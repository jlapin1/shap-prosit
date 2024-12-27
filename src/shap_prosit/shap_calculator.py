import logging
import os
import sys
from typing import Union

import pandas as pd
import numpy as np
import yaml
from numpy.typing import NDArray
import shap
sys.path.append(os.getcwd())
from src.models.model_wrappers import ModelWrapper, model_wrappers

class ShapCalculator:
    def __init__(
        self,
        mode: str,
        dset: NDArray,
        bgd: NDArray,
        model_wrapper: ModelWrapper,
        max_sequence_length: int = 30,
        max_charge: int = 6,
    ):
        self.val = dset
        self.bgd = bgd
        self.max_len = max_sequence_length
        self.max_charge = max_charge
        self.model_wrapper = model_wrapper

        self.bgd_sz = bgd.shape[0]

        if mode in {"rt", "cc", "charge1", "charge2", "charge3", "charge4", "charge5", "charge6"}:
            self.ext = 0
        else:
            self.ext = int(mode[1:].split("+")[0].split('^')[0])
        self.mode = mode

        self.fnull = np.array(
            [self.model_wrapper.make_prediction(bgd).squeeze().mean()]
        )

        self.savepep = []
        self.savecv = []

    def mask_pep(self, zs, pep, bgd_inds, mask=True) -> NDArray:
        BS, SL = zs.shape
        """
        With out specifying the data type, np.array(SL*['']) is automatically initialized
        as dtype 'U1'. This array dtype silently truncated strings down to their first
        character, which was an issue for modified amino acid strings.
        """
        out = np.tile(np.array(SL*[''], dtype='U12')[None], [BS,1])
        if mask:
            
            ## Collect all peptide tokens that are 'on' and place them in the out tensor
            oneinds = np.where(zs == 1)
            if len(oneinds[0]) > 0:
                out[oneinds] = np.tile(pep, [BS, 1])[oneinds] # == out[oneinds] = pep[oneinds[1]]
            
            ## Replace all peptide tokens that are 'off' with background dataset
            zeroinds = np.where(zs == 0)
            if len(zeroinds[0]) > 0:
                bgd_ = self.bgd[bgd_inds] # background dataset from batch_indices
                out[zeroinds] = bgd_[zeroinds]
            
            # pad c terminus with b''
            inds2 = (out=='').argmax(1)
            blanks = np.tile(np.arange(self.max_len)[None], [BS, 1]) >= inds2[:, None]
            out[:,:self.max_len][blanks] = ''
            
            # TODO: Consider randomly elongating peptides if bgd example is longer
            # TODO: Consider randomly turning truncated peptides into tryptic peptides

        else:
            out = pep

        # self.savepep.append(out)
        # self.savecv.append(zs)
        return out

    def ens_pred(self, pep, batsz=1000, mask=True):
        # pep: coalition vectors, 1s and 0s; excludes absent AAs
        P = pep.shape

        # Chunk into batches, each <= batsz
        batches = (
            np.split(pep, np.arange(batsz, batsz * (P[0] // batsz), batsz), 0)
            if P[0] % batsz == 0
            else np.split(pep, np.arange(batsz, batsz * (P[0] // batsz) + 1, batsz), 0)
        )

        # Use these indices to substitute values from background dataset
        # - bgd sample is run for each coalition vector
        rpts = P[0] // self.bgd_sz + 1  # number of repeats
        bgd_indices = np.concatenate(rpts * [np.arange(self.bgd_sz, dtype=np.int32)], axis=0)

        out_ = []
        for I, batch in enumerate(batches):
            # AAs (cut out CE, charge)
            # Absent AAs (all 1s)
            # [CE, CHARGE]
            batch = np.concatenate(
                [
                    batch[:, :-2],
                    np.ones((batch.shape[0], self.max_len - pep.shape[1] + 2)),
                    batch[:, -2:],
                ],
                axis=1,
            )
            #batch = th.tensor(batch, dtype=th.int32, device=device)

            # Indices of background dataset to use for subbing in 0s
            bgd_inds = bgd_indices[I * batsz : (I + 1) * batsz][: batch.shape[0]]

            # Create 1/0 mask and then turn into model ready input
            inp = self.mask_pep(batch, self.inp_orig, bgd_inds, mask)

            # Run through model
            out = self.model_wrapper.make_prediction(inp)
            out_.append(out)

        out_ = np.concatenate(out_, axis=0)

        return out_

    def score(self, peptide, mask=True):
        shape = peptide.shape
        x_ = self.ens_pred(peptide, mask=mask)
        score = x_.squeeze()
        if shape[0] == 1:
            score = np.array([score])[None, :]

        return score

    def calc_shap_values(self, sequence, samp=1000):
        # String array
        inp_orig = sequence
        self.inp_orig = inp_orig

        # Peptide length for the current peptide
        pl = sum(inp_orig[0] != "") - 2
        if pl <= self.ext:
            return False

        # Input coalition vector: All aa's on (1) + charge + eV
        # - Padded amino acids are added in as all ones (always on) in ens_pred
        inpvec = np.ones((1, pl + 2))

        # Mask vector is peptide length all off
        # - By turning charge and eV on, I am ignoring there contribution
        maskvec = np.zeros((self.bgd_sz, pl + 2))
        maskvec[:, -2:] = 1

        orig_spec = self.ens_pred(inpvec, mask=False)

        # SHAP Explainer
        ex = shap.KernelExplainer(self.score, maskvec)
        ex.fnull = self.fnull
        ex.expected_value = ex.fnull

        # Calculate the SHAP values
        seq = list(inp_orig.squeeze())
        seqrep = seq[:pl]
        # print(seqrep)

        inten = float(orig_spec.squeeze())

        shap_values = ex.shap_values(inpvec, nsamples=samp)

        return {
            "intensity": inten,
            "shap_values": shap_values.squeeze()[:pl],
            "sequence": seqrep,
            "energy": float(seq[-2]),
            "charge": int(seq[-1]),
        }


def save_shap_values(
    val_data_path: Union[str, bytes, os.PathLike],
    model_wrapper: ModelWrapper,
    mode: str,
    output_path: Union[str, bytes, os.PathLike] = ".",
    perm_path: Union[str, bytes, os.PathLike] = "perm.txt",
    samp: int = 1000,
    bgd_sz: int = 100,
):

    # Shuffle validation dataset and split it in background and validation.
    val_data = np.array([m.split(",") for m in open(val_data_path).read().split("\n")])
    if perm_path is None:
        perm = np.random.permutation(np.arange(len(val_data)))
        np.savetxt(output_path + "/perm.txt", perm, fmt="%d")
    else:
        perm = np.loadtxt(perm_path).astype(int)
    bgd = val_data[perm[:bgd_sz]]
    val = val_data[perm[bgd_sz:]]

    sc = ShapCalculator(mode, val, bgd, model_wrapper=model_wrapper)

    bgd_pred = model_wrapper.make_prediction(bgd)
    bgd_mean = np.mean(bgd_pred)

    result = {
        "sequence": [],
        "shap_values": [],
        "intensity": [],
        "energy": [],
        "charge": [],
        "bgd_mean": [],
    }
    # PUT IT BACK AFTER USAGE
    #for INDEX in range(1000):
    for INDEX in range(val.shape[0]):
        print("\r%d/%d" % (INDEX, len(val)), end="\n")
        sequence = sc.val[INDEX : INDEX + 1]
        out_dict = sc.calc_shap_values(sequence, samp=samp)
        if out_dict != False:
            for key, value in result.items():
                if key == "bgd_mean":
                    value.append(bgd_mean)
                else:
                    value.append(out_dict[key])
    pd.DataFrame(result).to_parquet(
        output_path + "/output.parquet.gzip", compression="gzip"
    )


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as file:
        config = yaml.safe_load(file)["shap_calculator"]

    if not os.path.exists(config["mode"]):
        os.makedirs(config["mode"])

    model_wrapper = model_wrappers[config["model_type"]](
        model_path=config["model_path"],
        ion_dict_path=config['ion_dict_path'],
        token_dict_path=config['token_dict_path'],
        yaml_dir_path=config['yaml_dir_path'],
        ion=config["ion"],
    )

    save_shap_values(
        val_data_path=config["val_inps_path"],
        model_wrapper=model_wrapper,
        mode=config["mode"],
        perm_path=config["perm_path"],
        output_path=config["mode"],
        samp=config["samp"],
        bgd_sz=config["bgd_sz"],
    )

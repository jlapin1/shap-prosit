import logging
import os
import sys
from typing import Union, List

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
        dset: pd.DataFrame,
        bgd: pd.DataFrame,
        model_wrapper: ModelWrapper,
        max_sequence_length: int = 30,
        max_charge: int = 6,
        inputs_ignored: int = 3,
    ):
        self.val = dset
        self.bgd = bgd
        self.max_len = max_sequence_length
        self.max_charge = max_charge
        self.model_wrapper = model_wrapper
        self.inputs_ignored = inputs_ignored

        self.bgd_size = bgd.shape[0]

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
        shape = pep.shape

        # Chunk into batches, each <= batsz
        batches = (
            np.split(pep, np.arange(batsz, batsz * (shape[0] // batsz), batsz), 0)
            if shape[0] % batsz == 0
            else np.split(pep, np.arange(batsz, batsz * (shape[0] // batsz) + 1, batsz), 0)
        ) # -> List

        # Use these indices to substitute values from background dataset
        # - bgd sample is run for each coalition vector
        rpts = shape[0] // self.bgd_size + 1  # number of repeats
        bgd_indices = np.concatenate(rpts * [np.arange(self.bgd_size, dtype=np.int32)], axis=0)

        out_ = []
        for I, batch in enumerate(batches):
            # AAs (cut out CE, charge)
            # Absent AAs (all 1s)
            # [CE, CHARGE]
            batch = np.concatenate(
                [
                    batch[:, :-self.inputs_ignored],
                    np.ones((batch.shape[0], self.max_len - shape[1] + self.inputs_ignored)),
                    batch[:, -self.inputs_ignored:],
                ],
                axis=1,
            )
            #batch = th.tensor(batch, dtype=th.int32, device=device)

            # Indices of background dataset to use for subbing in 0s
            bgd_inds = bgd_indices[I * batsz : (I + 1) * batsz][: batch.shape[0]]

            # Create 1/0 mask and then turn into model ready input
            inp = self.mask_pep(batch, self.input_orig, bgd_inds, mask)

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
        input_orig = sequence
        self.input_orig = input_orig

        # Peptide length for the current peptide
        num_ignored = self.inputs_ignored
        peptide_length = sum(input_orig[0, :-num_ignored] != '')
        shap_vector_length = peptide_length + num_ignored
        if peptide_length <= self.ext:
            return False

        # Input coalition vector: All aa's on (1) + charge + eV
        # - Padded amino acids are added in as all ones (always on) in ens_pred
        inpvec = np.ones((1, shap_vector_length))

        # Mask vector is peptide length all off
        # - By turning the ignored inputs on, I am ignoring there contribution
        maskvec = np.zeros((self.bgd_size, shap_vector_length))
        maskvec[:, -num_ignored: ] = 1

        orig_spec = self.ens_pred(inpvec, mask=False)

        # SHAP Explainer
        ex = shap.KernelExplainer(self.score, maskvec)
        ex.fnull = self.fnull
        ex.expected_value = ex.fnull

        # Calculate the SHAP values
        seq = list(input_orig.squeeze())
        seqrep = seq[:peptide_length]
        inten = float(orig_spec.squeeze())
        shap_values = ex.shap_values(inpvec, nsamples=samp, silent=True)

        # TODO Find a dynamic way of including arbitrary number non-sequence items

        return {
            "intensity": inten,
            "shap_values": shap_values.squeeze()[:peptide_length],
            "sequence": seqrep,
            "charge": int(seq[-3]),
            "energy": float(seq[-2]),
            "method": seq[-1],
        }

def save_shap_values(
    val_data_path: Union[str, bytes, os.PathLike],
    model_wrapper: ModelWrapper,
    mode: str,
    output_path: Union[str, bytes, os.PathLike] = ".",
    perm_path: Union[str, bytes, os.PathLike] = "perm.txt",
    samp: int = 1000,
    bgd_size: int = 100,
    inputs_ignored: int = 3,
    queries: List[str] = None,
):
    print("<<<ATTN>>> Starting calculation loop")

    # Load and query data
    val_data = pd.read_parquet(val_data_path)
    original_size = val_data.shape[0]
    if queries is not None:
        query_expression = " and ".join(queries)
        print(f"<<<ATTN>>> Querying dataset of size {original_size} with expression: '{query_expression}'")
        val_data = val_data.query(query_expression)
        new_size = val_data.shape[0]
        print(f"<<<ATTN>>> Dataset now has size {val_data.shape[0]}")
        if (new_size == original_size) or (new_size == 0):
            print("<<<ATTN>>> WARNING query didn't do anything, or it did too much")
    
    # Shuffle validation dataset and split it in background and validation.
    if perm_path is None:
        perm = np.random.permutation(np.arange(len(val_data)))
    else:
        perm = np.loadtxt(perm_path).astype(int)
    np.savetxt(output_path + "/perm.txt", perm, fmt="%d")
    bgd = np.stack(val_data.iloc[perm[:bgd_size]]['full'])
    val = np.stack(val_data.iloc[perm[bgd_size:]]['full'])

    sc = ShapCalculator(
        mode, 
        val, 
        bgd, 
        model_wrapper=model_wrapper,
        inputs_ignored=inputs_ignored,
    )

    bgd_pred = model_wrapper.make_prediction(bgd)
    bgd_mean = np.mean(bgd_pred)
    
    # TODO arbitrary number of non-sequence items
    result = {
        "sequence": [],
        "shap_values": [],
        "intensity": [],
        "energy": [],
        "charge": [],
        "method": [],
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
        
        # Dump results every 1000 explanations to be safe
        if (INDEX+1) % 100 == 0:
            pd.DataFrame(result).to_parquet(
                output_path + "/output.parquet", compression="gzip"
            )
    
    pd.DataFrame(result).to_parquet(
        output_path + "/output.parquet", compression="gzip"
    )


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as file:
        config = yaml.safe_load(file)["shap_calculator"]
    
    output_dir = config["mode"] if config['output_dir'] is None else config['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system(f"cp config.yaml {output_dir}/")

    model_wrapper = model_wrappers[config["model_type"]](
        model_path=config["model_path"],
        ion_dict_path=config['ion_dict_path'],
        token_dict_path=config['token_dict_path'],
        yaml_dir_path=config['yaml_dir_path'],
        mode=config["mode"],
        method_list=config['method_list'],
    )

    save_shap_values(
        val_data_path=config["val_inps_path"],
        queries=config['queries'],
        model_wrapper=model_wrapper,
        mode=config["mode"],
        perm_path=config["perm_path"],
        output_path=output_dir,
        samp=config["samp"],
        bgd_size=config["bgd_sz"],
        inputs_ignored=config['inputs_ignored'],
    )

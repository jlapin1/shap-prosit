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
from tqdm import tqdm

class ShapCalculator:
    def __init__(
        self,
        mode: List[str],
        dset: pd.DataFrame,
        bgd: pd.DataFrame,
        model_wrapper: ModelWrapper,
        batch_size: int = 1000,
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
        self.batch_size = batch_size

        self.bgd_size = bgd.shape[0]

        if mode[0] in ["rt", "cc", "fly1", "fly2", "fly3", "fly4",
                       "charge1", "charge2", "charge3", "charge4", "charge5", "charge6"]:
            self.ext = 0
        else:
            self.ext = {ion: int(ion[1:].split("+")[0].split('^')[0]) for ion in mode}
        self.mode = mode
        self.fnull = self.model_wrapper.make_prediction(bgd).mean(0)

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
        x_ = self.ens_pred(peptide, self.batch_size, mask=mask)
        score = x_
        #if shape[0] == 1:
        #    score = np.array([score])[None, :]

        return score

    def calc_shap_values(self, sequence, samp=1000):
        # String array
        input_orig = sequence
        self.input_orig = input_orig

        # Peptide length for the current peptide
        num_ignored = self.inputs_ignored
        peptide_length = sum(input_orig[0, :-num_ignored] != '')
        shap_vector_length = peptide_length + num_ignored

        # Input coalition vector: All aa's on (1) + charge + eV
        # - Padded amino acids are added in as all ones (always on) in ens_pred
        inpvec = np.ones((1, shap_vector_length))

        # Mask vector is peptide length all off
        # - By turning the ignored inputs on, I am ignoring there contribution
        maskvec = np.zeros((self.bgd_size, shap_vector_length))
        maskvec[:, -num_ignored: ] = 1

        # SHAP Explainer
        ex = shap.KernelExplainer(self.score, maskvec)#, keep_index=True)
        ex.fnull = self.fnull
        ex.expected_value = ex.fnull

        # Calculate the SHAP values
        shap_values = ex.shap_values(inpvec, nsamples=samp, silent=True)
        
        # Other outputs to save
        seq = list(input_orig.squeeze())
        seqrep = seq[:peptide_length]
        original_intensity = self.ens_pred(inpvec, self.batch_size, mask=False)
        too_short = False if self.ext == 0 else peptide_length <= np.array(list(self.ext.values()))
        # Identify impossible sequences by setting intensity to -1
        original_intensity[:, too_short] = -1

        shap_values = np.array(shap_values)
        if self.ext == 0:
            shap_values = shap_values.reshape(1, -1, len(self.mode))

        # TODO Find a dynamic way of including arbitrary number non-sequence items

        return {
            "intensity": pd.Series(original_intensity.squeeze(), index=self.mode),
            "shap_values": pd.DataFrame(shap_values[0, :peptide_length], columns=self.mode),
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
    bgd_loc_path: Union[str, bytes, os.PathLike] = None,
    base_samp: int = 1000,
    extra_samp: List[int] = None,
    bgd_size: int = 100,
    inputs_ignored: int = 3,
    dataset_queries: List[str] = None,
    bgd_queries: List[str] = None,
    batch_size: int = 1000,
    **kwargs
):
    print("<<<ATTN>>> Starting calculation loop")

    # Load and query data
    val_data = pd.read_parquet(val_data_path)
    original_size = val_data.shape[0]
    if dataset_queries is not None:
        query_expression = " and ".join(dataset_queries)
        print(f"<<<ATTN>>> Querying dataset of size {original_size} with expression: '{query_expression}'")
        val_data = val_data.query(query_expression)
        new_size = val_data.shape[0]
        print(f"<<<ATTN>>> Dataset now has size {val_data.shape[0]}")
        if (new_size == original_size) or (new_size == 0):
            print("<<<ATTN>>> WARNING query didn't do anything, or it did too much")
        print(val_data)
    
    # Split dataset into background and validation.
    # Existing split
    if bgd_loc_path is not None:
        print("<<<ATTN>>> Loading existing bgd split")
        loc_inds = np.loadtxt(bgd_loc_path).astype(int)
        bgd = val_data.loc[loc_inds]
    
    # Must create a new split
    else:
        print("<<<ATTN>>> Creating new bgd split")
        if bgd_queries is not None:
            query_expression = " and ".join(bgd_queries)
            print(f"<<<ATTN>>> Querying bgd dataset with expression: '{query_expression}'")
            bgd = val_data.query(query_expression)
        else:
            bgd = val_data
        bgd = bgd.sample(bgd_size)
    
    bgd_indices = bgd.index.values.tolist()
    np.savetxt(output_path + "/bgd_loc_indices.txt", bgd_indices, fmt="%d")
    remaining_indices = val_data.index.values.tolist()
    for index in bgd_indices: remaining_indices.remove(index)
    np.savetxt(output_path + "/val_loc_indices.txt", remaining_indices, fmt='%d')
    bgd = np.stack(bgd['full'])
    val = np.stack(val_data.loc[remaining_indices]['full'])

    sc = ShapCalculator(
        mode, 
        val, 
        bgd,
        model_wrapper=model_wrapper,
        batch_size=batch_size,
        inputs_ignored=inputs_ignored,
    )
    
    bgd_mean = pd.Series(sc.fnull, index=sc.mode)
    
    # TODO arbitrary number of non-sequence items
    result = {
        "sequence": [],
        "energy": [],
        "charge": [],
        "method": [],
    }
    for mode_ in sc.mode:
        result[f'bgd_mean_{mode_}'] = []
        result[f'intensity_{mode_}'] = []
        result[f'shap_values_{mode_}'] = []
    
    pbar = tqdm(range(val.shape[0]))
    for INDEX in pbar:
        pbar.set_description("Calculating SHAP explanations")
        sequence = sc.val[INDEX : INDEX + 1]
        
        # Set sampling amount
        if extra_samp is not None:
            explain_length = sum(sequence.squeeze()[:-inputs_ignored] != '')
            if explain_length >= extra_samp[0]:
                Samp = extra_samp[1]
            else:
                Samp = base_samp
        else:
            Samp = base_samp

        # Calculate shapley values
        out_dict = sc.calc_shap_values(sequence, samp=Samp)
        
        # Save results
        if out_dict != False:
            for key, value in result.items():
                if "bgd_mean" in key:
                    mode = key.split('_')[-1]
                    value.append(bgd_mean[mode])
                elif 'intensity' in key:
                    mode = key.split('_')[-1]
                    value.append(out_dict['intensity'][mode])
                elif 'shap' in key:
                    mode = key.split('_')[-1]
                    value.append(out_dict['shap_values'][mode].to_list())
                else:
                    value.append(out_dict[key])
        
        # Dump results every 100 explanations to be safe
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
    
    # Output directory
    config_ = config['shap_settings']
    output_dir = config_["mode"] if config_['output_dir'] is None else config_['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system(f"cp config.yaml {output_dir}/")
    
    # Model
    model_wrapper = model_wrappers[config['model_settings']["model_type"]](
        mode=config['shap_settings']["mode"],
        **config['model_settings'],
    )
    
    # SHAP calculation
    save_shap_values(
        model_wrapper=model_wrapper,
        output_path=output_dir,
        **config['shap_settings']
    )

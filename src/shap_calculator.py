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
import src.utils as U
from tqdm import tqdm
import re
DTYPE = "<U30"

class ShapCalculator:
    def __init__(
        self,
        dset: pd.DataFrame,
        bgd: pd.DataFrame,
        model_wrapper: ModelWrapper,
        peptides: list = None,
        batch_size: int = 1000,
        max_input_length: int = 50,
        outputs: int = 40,
        max_charge: int = 6,
        inputs_ignored: int = 2,
        blank_token: int = 0,
    ):
        self.val = dset
        self.bgd = bgd
        self.max_len = max_input_length
        self.num_outputs = outputs + 1 
        self.max_charge = max_charge
        self.model_wrapper = model_wrapper
        self.inputs_ignored = inputs_ignored
        self.batch_size = batch_size
        self.blank_token = blank_token
        self.dtype = DTYPE

        self.bgd_size = bgd.shape[0]

        self.mode = np.arange(self.num_outputs) # explanations for amino acids from 1 - num_outputs
        self.fnull = np.zeros((self.num_outputs,)) #self.model_wrapper.make_prediction(bgd).mean(0)

        self.process_spectra()
        if peptides is not None:
            self.process_peptide_sequences(peptides)

    def process_spectra(self):
        entire_dataset = np.full(( len(self.val), 2, self.max_len + self.inputs_ignored ), '0', dtype=self.dtype)
        pbar = tqdm(self.val.items(), total=len(self.val))
        for iloc, (loc, linear_input) in enumerate(pbar):
            pbar.set_description("Processing dataset into numpy tensors")
            spectrum = linear_input[:-self.inputs_ignored].reshape(2, -1)
            other = linear_input[-self.inputs_ignored:]
            
            # Top n peaks
            spectrum = U.sort_array(spectrum, 1, descending=True, first=50)
            spectrum = U.sort_array(spectrum, 0)
            
            # Place sorted spectrum inside new variable
            actual_length = spectrum.shape[1]
            tensor = np.full((2, self.max_len + self.inputs_ignored), '0', dtype=self.dtype)
            tensor[:, :actual_length] = spectrum
            tensor[0, -self.inputs_ignored:] = other

            entire_dataset[iloc] = tensor
        
        self.val = entire_dataset

    def process_peptide_sequences(self, peptides):
        self.answer = {
            'modseq': [],
            'aaseq': [],
            'intseq': [],
            #'matched_ions': [],
            #'matched_mzs': [],
            #'matched_inds': [],
        }
        pbar = tqdm(peptides)
        for i, peptide in enumerate(pbar):
            pbar.set_description("Processing peptide sequences")
            tokenized_sequence = model_wrapper.D.data.tokenizer(peptide)
            intseq = [self.model_wrapper.D.data.amod_dic[m] for m in tokenized_sequence]
            
            #matched_ions, matched_mzs, matched_inds = U.match(peptide, int(self.val[i, 0, -2]), self.val[i, 0, :-self.inputs_ignored])

            self.answer['modseq'].append(peptide)
            self.answer['aaseq'].append(tokenized_sequence)
            self.answer['intseq'].append(intseq)
            #self.answer['matched_ions'].append(matched_ions)
            #self.answer['matched_mzs'].append(matched_mzs)
            #self.answer['matched_inds'].append(matched_inds)

        self.answer = pd.DataFrame(self.answer)

    def mask_pep(self, zs, pep, bgd_inds, mask=True) -> NDArray:
        BS, SL = zs.shape
        """
        With out specifying the data type, np.array(SL*[blank]) is automatically initialized
        as dtype 'U1'. This array dtype silently truncated strings down to their first
        character, which was an issue for modified amino acid strings.
        """
        #out = np.tile(np.array(2, SL*[self.blank_token], dtype=np.float32)[None], [BS,1])
        out = np.tile(np.array(SL*[self.blank_token], dtype=DTYPE)[None, None], [BS,2,1])
        zsexp = np.tile(zs[:,None], [1,2,1])
        if mask:
            
            ## Collect all peptide tokens that are 'on' and place them in the out tensor
            oneinds = np.where(zsexp == 1) # np.tile(zs[:,None], [1,2,1]) == 1
            if len(oneinds[0]) > 0:
                out[oneinds] = np.tile(pep, [BS, 1, 1])[oneinds] # == out[oneinds] = pep[oneinds[1]]
            
            ## Replace all peptide tokens that are 'off' with background dataset
            zeroinds = np.where(zsexp == 0)
            if len(zeroinds[0]) > 0:
                bgd_ = self.bgd[bgd_inds] # background dataset from batch_indices
                bgd_ = np.tile(bgd_[:,None], [1,2,1])
                out[zeroinds] = bgd_[zeroinds]
            
            # Place new null peaks at the end of the sequence (before ignored inputs)
            out[:,:,:-self.inputs_ignored] = U.sort_array(out[:,:,:-self.inputs_ignored], 0, sub_values=[0, 1e9])
            
        else:
            out = pep

        # self.savepep.append(out)
        # self.savecv.append(zs)
        return out

    def ens_pred(self, spec, batsz=1000, mask=True, silent=False):
        # pep: coalition vectors, 1s and 0s; excludes absent AAs
        shape = spec.shape
        if shape[0] == 1: silent=True

        # Chunk into batches, each <= batsz
        batches = (
            np.split(spec, np.arange(batsz, batsz * (shape[0] // batsz), batsz), 0)
            if shape[0] % batsz == 0
            else np.split(spec, np.arange(batsz, batsz * (shape[0] // batsz) + 1, batsz), 0)
        ) # -> List

        # Use these indices to substitute values from background dataset
        # - bgd sample is run for each coalition vector
        rpts = shape[0] // self.bgd_size + 1  # number of repeats
        bgd_indices = np.concatenate(rpts * [np.arange(self.bgd_size, dtype=np.int32)], axis=0)

        out_ = []
        pbar = batches if silent==True else tqdm(batches)
        for I, batch in enumerate(pbar):
            if not silent: pbar.set_description("ens_pred loop")
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
            out = self.model_wrapper.make_prediction(inp, target=self.target)
            out_.append(out.cpu().numpy())

        out_ = np.concatenate(out_, axis=0)

        return out_

    def score(self, spectrum, mask=True):
        shape = spectrum.shape
        x_ = self.ens_pred(spectrum, self.batch_size, mask=mask)
        score = x_
        
        return score

    def calc_shap_values(self, sequence, samp=1000, **kwargs):
        # String array
        input_orig = sequence
        self.input_orig = input_orig

        # spectrum length for the current peptide
        num_ignored = self.inputs_ignored
        spectrum_length = sum(input_orig[0, 0, :-num_ignored] != str(self.blank_token))
        shap_vector_length = spectrum_length + num_ignored

        # Input coalition vector: All peaks on (1) + charge + mass
        # - Padded peaks are added in as all ones (always on) in ens_pred
        inpvec = np.ones((1, shap_vector_length))
        
        # Get model's predicted peptide 
        # - Reminder: diffusion models are non-deterministic
        max_length = len(kwargs['peptide'])+1 if 'peptide' in kwargs else None
        predicted_aa_list, predicted_intseq = self.model_wrapper.predict_peptide(self.input_orig, max_length=max_length)
        self.target = predicted_intseq[None]
        #print(f"Predicted peptide: {''.join(predicted_aa_list)}")
        if 'peptide' in kwargs:
            correct = re.sub('I', 'L', ''.join(predicted_aa_list)) == re.sub('I', 'L', "".join(kwargs['peptide']))
            if not correct:
                return False
        
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
        shap_values = np.array(shap_values[0, :-num_ignored])
        shap_values_ = np.zeros(shape=(shap_values.shape[0] ,len(self.mode)))
        shap_values_[:shap_values.shape[0], :shap_values.shape[1]] = shap_values

        return {
            "mz": input_orig[0, 0, :-num_ignored].astype(np.float32),
            "intensity": input_orig[0, 1, :-num_ignored].astype(np.float32),
            "charge": int(input_orig[0, 0, -4]),
            "mass": float(input_orig[0, 0, -3]),
            "fragmentation_method": input_orig[0, 0, -2],
            "enzyme": input_orig[0, 0, -1],
            "pred_aaseq": predicted_aa_list,
            "shap_values": pd.DataFrame(shap_values_, columns=self.mode),
        }

def save_shap_values(
    val_data_path: Union[str, bytes, os.PathLike],
    model_wrapper: ModelWrapper,
    output_path: Union[str, bytes, os.PathLike] = ".",
    bgd_loc_path: Union[str, bytes, os.PathLike] = None,
    base_samp: int = 1000,
    extra_samp: List[int] = None,
    bgd_size: int = 100,
    inputs_ignored: int = 2,
    max_peaks: int = 50,
    max_peptide_length = 40,
    dataset_queries: List[str] = None,
    bgd_queries: List[str] = None,
    batch_size: int = 1000,
    **kwargs
):
    print("<<<ATTN>>> Starting calculation loop")

    # Load data
    val_data = pd.read_parquet(val_data_path)
    original_size = val_data.shape[0]
    
    """
    # Load existing split BEFORE querying dataset
    if bgd_loc_path is not None:
        print("<<<ATTN>>> Loading existing bgd split")
        loc_inds = np.loadtxt(bgd_loc_path).astype(int)
        bgd = val_data.loc[loc_inds]
    
    # Query dataset
    if dataset_queries is not None:
        query_expression = " and ".join(dataset_queries)
        print(f"<<<ATTN>>> Querying dataset of size {original_size} with expression: '{query_expression}'")
        val_data = val_data.query(query_expression)
        new_size = val_data.shape[0]
        print(f"<<<ATTN>>> Dataset now has size {val_data.shape[0]}")
        if (new_size == original_size) or (new_size == 0):
            print("<<<ATTN>>> WARNING query didn't do anything, or it did too much")
        print(val_data)
    
    # Create a new split (if not loading existing)
    if bgd_loc_path is None:
        print("<<<ATTN>>> Creating new bgd split")
        if bgd_queries is not None:
            query_expression = " and ".join(bgd_queries)
            print(f"<<<ATTN>>> Querying bgd dataset with expression: '{query_expression}'")
            bgd = val_data.query(query_expression)
        else:
            bgd = val_data
        bgd = bgd.sample(bgd_size)
    
    # Save splits
    bgd_indices = bgd.index.values.tolist()
    np.savetxt(output_path + "/bgd_loc_indices.txt", bgd_indices, fmt="%d")
    remaining_indices = val_data.index.values.tolist()
    for index in bgd_indices: 
        try:
            remaining_indices.remove(index)
        except:
            # This can happen if you load an existing bgd split, but querying
            # the dataset get rid of those bgd loc indices
            pass
    np.savetxt(output_path + "/val_loc_indices.txt", remaining_indices, fmt='%d')
    
    # Convert full column to numpy arrays
    bgd = np.stack(bgd['full'])
    val = np.stack(val_data.loc[remaining_indices]['full'])
    """
    val = val_data['full'].map(lambda x: np.array(x.split(','), dtype=DTYPE))
    peptides = val_data['modified_sequence'].to_list()
    tokenized = val_data['modified_sequence'].map(lambda x: model_wrapper.D.data.tokenizer(x)).to_list()
    bgd = np.full((1, max_peaks), model_wrapper.blank_token, dtype=DTYPE)

    # NOTE: sequence length can be different than peptide length
    max_input_length = max_peaks #val.shape[1] - inputs_ignored

    sc = ShapCalculator(
        val, 
        bgd,
        peptides=peptides,
        model_wrapper=model_wrapper,
        batch_size=batch_size,
        inputs_ignored=inputs_ignored,
        max_input_length=max_input_length,
        outputs=max_peptide_length,
    )
    
    bgd_mean = pd.Series(sc.fnull, index=sc.mode)
    
    # TODO arbitrary number of non-sequence items
    result = {}
    
    pbar = tqdm(range(0, val.shape[0], 1))
    for INDEX in pbar:
        pbar.set_description("Calculating SHAP explanations")
        sequence = sc.val[INDEX : INDEX + 1]
        
        # Set sampling amount
        Samp = base_samp

        # Calculate shapley values
        out_dict = sc.calc_shap_values(sequence, samp=Samp, peptide=tokenized[INDEX])
        if out_dict == False:
            continue
        # add to out_dict to include in output
        
        # Create sparse arrays for shapley values by mode
        shap_results = {}
        shap_values = out_dict.pop("shap_values")
        for column in shap_values:
            ind_col_name = f"sv_indices_{column}"
            mode_indices = np.where(shap_values[column] != 0)[0].astype(np.int16)
            shap_results[ind_col_name] = mode_indices
            sv_col_name = f"sv_values_{column}"
            mode_shap_values = shap_values[column].iloc[mode_indices].to_numpy().astype(np.float32)
            shap_results[sv_col_name] = mode_shap_values
        
        # Add answer data
        addons = {}
        if hasattr(sc, 'answer'):
            answer_dict = sc.answer.iloc[INDEX].to_dict()
            for key in answer_dict:
                addons[f"answer_{key}"] = answer_dict[key]
        else:
            addons = {}


        new_dict = out_dict | shap_results | addons

        # Save results
        if new_dict != False:
            for key, value in new_dict.items():
                if key not in result:
                    result[key] = []
                result[key].append(value)
        
        # Dump results every 100 explanations to be safe
        if (INDEX+1) % 1 == 0:
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
    os.system(f"cp {sys.argv[1]} {output_dir}/")
    
    # Model
    model_type = 'koina' if 'koina' in config['model_settings']['model_type'] else 'local'
    model_wrapper = model_wrappers[config['model_settings']["model_type"]](
        ignored_inputs=config['shap_settings']['inputs_ignored'],
        **config['model_settings'][model_type],
    )
    
    # SHAP calculation
    save_shap_values(
        model_wrapper=model_wrapper,
        output_path=output_dir,
        **config['shap_settings']
    )

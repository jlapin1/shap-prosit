"""
All model wrappers must have a make_prediction function that:
- converts linearized string vectors to model ready inputs
- runs inputs through the model
- returns the output of interest

Converting the SHAP linearized numpy string array input 
e.g. [seq,charge,energy,method] to model ready outputs is done
using the function def hx(inputs).
"""

import os
from abc import ABC, abstractmethod
from time import sleep
from typing import Union
import warnings

import pandas as pd
import numpy as np
import re

import yaml
from numpy import ndarray

from koinapy import Koina

#from . import PeptideEncoder
#import torch as th
#device = th.device("cuda" if th.cuda.is_available() else "cpu")

#from . import TransformerModel
from .ChargeState import custom_keras_utils as cutils
from .ChargeState.chargestate import ChargeStateDistributionPredictor
from .ChargeState.dlomix_preprocessing import to_dlomix
import keras
import tensorflow as tf

def hx(tokens):
    sequence = tokens[:, :-2]
    collision_energy = tf.strings.to_number(tokens[:, -2:-1])
    precursor_charge = tf.one_hot(
        tf.cast(tf.strings.to_number(tokens[:, -1]), tf.int32) - 1, 6
    )
    z = {
        "sequence": sequence,
        "collision_energy": collision_energy,
        "precursor_charge": precursor_charge,
    }

    return z


def annotations(max_len: int = 30):
    """Generate dictionary with ion ids.`

    Returns:
        dict: Ion ids dictionary.
    """
    ions = {}
    count = 0
    for i in np.arange(1, max_len):
        for j in ["y", "b"]:
            for k in [1, 2, 3]:
                ion = f"{j}{i}+{k}"
                ions[ion] = count
                count += 1
    return ions

def integerize_sequence(seq, dictionary):
    return [dictionary[aa] for aa in seq]

class ModelWrapper(ABC):
    @abstractmethod
    def __init__(self, path: Union[str, bytes, os.PathLike], mode: str) -> None:
        """Initialize model wrapper by loading the model."""
        pass

    @abstractmethod
    def make_prediction(self, inputs: ndarray) -> ndarray:
        """Make prediction from input array containing
        sequences, precursor charges and collision energy."""
        pass

class PrositIntensityWrapper(ModelWrapper):
    def __init__(
        self, model_path: Union[str, bytes, os.PathLike], 
        mode: str
    ) -> None:

        self.ion_ind = annotations()[mode]
        self.model = PrositIntensityPredictor(seq_length=30)
        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        self.model.load_weights(latest_checkpoint)

    def make_prediction(self, inputs: ndarray) -> ndarray:
        return (self.model(hx(inputs), training=False)[:, self.ion_ind]).numpy()


class TransformerIntensityWrapper(ModelWrapper):
    def __init__(
        self, 
        model_path: Union[str, bytes, os.PathLike], 
        mode: str
    ) -> None:
        
        self.ion_ind = annotations()[mode]
        with open(os.path.join(model_path, "model.yaml"), encoding="utf-8") as file:
            model_config = yaml.safe_load(file)
        self.model = TransformerModel(**model_config)
        latest_checkpoint = tf.train.latest_checkpoint(path)
        self.model.load_weights(latest_checkpoint)
    
    def make_prediction(self, inputs: ndarray) -> ndarray:
        return (self.model(hx(inputs), training=False)[:, self.ion_ind]).numpy()

class TorchIntensityWrapper(ModelWrapper):
    def __init__(self, 
        model_path: Union[str, bytes, os.PathLike], 
        ion_dict_path: Union[str, bytes, os.PathLike],
        token_dict_path: Union[str, bytes, os.PathLike],
        yaml_dir_path: Union[str, bytes, os.PathLike],
        mode: str,
        method_list: list,
    ) -> None:

        ion_dict = pd.read_csv(ion_dict_path, index_col='full')
        self.ion_ind = ion_dict.loc[mode]['index']
        with open(os.path.join(yaml_dir_path, "model.yaml")) as f: model_config = yaml.safe_load(f)
        with open(os.path.join(yaml_dir_path, "loader.yaml")) as f: load_config = yaml.safe_load(f)
        num_tokens = len(open(token_dict_path).read().strip().split("\n")) + 1
        self.model = PeptideEncoder(
            tokens = num_tokens,
            final_units = len(ion_dict),
            max_charge = load_config['charge'][-1],
            **model_config
        )
        self.model.to(device)
        self.model.load_state_dict(th.load(model_path, map_location=device))
        self.model.eval()
        
        self.token_dict = self.create_dictionary(token_dict_path)

        # Same code as in loader_hf.py
        self.method_dic = {method: m for m, method in enumerate(method_list)}
        self.method_dicr = {n:m for m,n in self.method_dic.items()}

    def create_dictionary(self, dictionary_path):
        amod_dic = {
            line.split()[0]:m for m, line in enumerate(open(dictionary_path))
        }
        amod_dic['X'] = len(amod_dic)
        amod_dic[''] = amod_dic['X']
        return amod_dic

    def hx(self, linear_input):
        """
        Turn list of [seq, energy, charge, method] into model ready tensors
        """
        sequence = linear_input[:,:-3]
        intseq = th.tensor(
            [
                integerize_sequence(seq, self.token_dict)
                for seq in sequence
            ], 
            dtype=th.int32, device=device
        )
        charge = th.tensor(linear_input[:,-3].astype(int), dtype=th.int32, device=device)
        energy = th.tensor(linear_input[:,-2].astype(float), dtype=th.float32, device=device)
        method = th.tensor(
            [self.method_dic[tok] for tok in linear_input[:,-1]], 
            dtype=th.int32,
            device=device,
        )
        return {
            'intseq': intseq,
            'charge': charge,
            'energy': energy,
            'method': method,
        }

    def make_prediction(self, inputs: ndarray) -> ndarray:
        with th.no_grad():
            out = self.model(**self.hx(inputs))
        out = out / out.max(dim=1, keepdim=True)[0]
        prediction = (out[:, self.ion_ind]).detach().cpu().numpy()
        return prediction

class KoinaWrapper(ModelWrapper):
    def __init__(
        self, 
        model_path: Union[str, bytes, os.PathLike], 
        mode: str,
        inputs_ignored: int=3,
        **kwargs
    ) -> None:

        self.model = Koina(model_path)
        self.mode = mode
        self.inputs_ignored = inputs_ignored

    def make_prediction(self, inputs: ndarray, silent: bool=True) -> ndarray:
        # set mode = "rt" to enter retention time mode
        # inputs["sequences"] = [["W", "E"],[],[]]
        if self.mode == "rt":

            print(type(hx(inputs)["sequence"][0]))

            sequences = []
            for seq in hx(inputs)["sequence"]:
                try:
                    sequence = "".join(seq)
                except:
                    sequence = b"".join(seq).decode("utf-8")
                sequences.append(sequence)

            input_df = pd.DataFrame()
            input_df["peptide_sequences"] = np.array(sequences)

            return (self.model.predict(input_df)["irt"]).to_numpy()

        elif self.mode == "cc":
            # set mode = "cc" to enter retention time mode
            sequences = []
            for seq in hx(inputs)["sequence"]:
                try:
                    sequence = "".join(seq)
                except:
                    sequence = b"".join(seq).decode("utf-8")
                sequences.append(sequence)

            input_df = pd.DataFrame()
            input_df["peptide_sequences"] = np.array(sequences)
            input_df["precursor_charges"] = inputs[:, -1].astype("int")
            print(input_df["peptide_sequences"])
            print(input_df["precursor_charges"])
            assert len(input_df.shape) == 2, f"shape is not 2-dimensional\n{input_df}"
            return (self.model.predict(input_df)["ccs"]).to_numpy()

        else:
            sequences = []
            for i in inputs[:, :-self.inputs_ignored]:
                try:
                    sequence = "".join(i)
                except:
                    sequence = b"".join(i).decode("utf-8")
                # FIXME If sequence contains mods that prosit can't handle, remove the input
                # instance
                sequence = re.sub('\[UNIMOD:4]', '', sequence)
                sequence = re.sub('\[UNIMOD:1]', '', sequence)
                sequences.append(sequence)
            # FIXME This always assumes charge is third to last and collision energy second
            # to last (method is last). Need a dynamic way of finding these two features'
            # positions.
            input_dict = {
                "peptide_sequences": np.array(sequences),
                "precursor_charges": inputs[:, -3].astype("int"),
                "collision_energies": (inputs[:, -2].astype("float")).astype(
                    "int"
                ),
            }
            counter = 0
            success = False
            while counter < 5 and not success:
                try:
                    preds = self.model.predict(
                        pd.DataFrame(input_dict), min_intensity=-0.00001
                    )
                except:
                    print(input_dict)
                    counter += 1
                    sleep(1)
                else:
                    success = True
            if counter >= 5:
                return np.zeros(len(inputs))
            
            # Find the annotation/mode in the koina output
            ann_bool = preds["annotation"] == bytes(self.mode, "utf-8")
            
            # If you don't find it, return 0 prediction
            if len(preds[ann_bool]["intensities"]) < len(sequences):
                if silent == False: print(preds)
                results = []
                for i in range(len(sequences)):
                    if i not in preds[ann_bool]["intensities"].index:
                        results.append(0.0)
                    else:
                       results.append(preds[ann_bool]["intensities"][i])
            else:
                results = preds[ann_bool]["intensities"].to_numpy()
            return np.array(results)


class ChargeStateWrapper(ModelWrapper):
    def __init__(
        self, 
        model_path: Union[str, bytes, os.PathLike], 
        mode: str
    ) -> None:

        import ChargeState

        self.path = "./src/models/ChargeState/trained_model.keras"
        if path:
            warnings.warn(
                """
                            You have given a path even though this project provides a model already.
                            If provided a .keras model in the same form as the provided model,
                            the code might still work. Otherwise, stick with provided model or modify this code section.
                            """
            )
            self.path = model_path
        self.mode = mode[-1]

        self.model = keras.saving.load_model(
            self.path,
            custom_objects={
                "upscaled_mean_squared_error": cutils.upscaled_mean_squared_error,
                "euclidean_similarity": cutils.euclidean_similarity,
                "masked_pearson_correlation_distance": cutils.masked_pearson_correlation_distance,
                "masked_spectral_distance": cutils.masked_spectral_distance,
                "ChargeStateDistributionPredictor": ChargeStateDistributionPredictor,
            },
        )

    def make_prediction(self, inputs: ndarray) -> ndarray:

        # inputs["sequences"] = [["W", "E"],[],[]]

        # print(type(hx(inputs)["sequence"][0]))

        sequences = []
        for seq in hx(inputs)["sequence"]:
            try:
                sequence = "".join(seq)
            except:
                sequence = b"".join(seq).decode("utf-8")
            sequences.append(sequence)

        input_df = pd.DataFrame()
        input_df["peptide_sequences"] = np.array(sequences)

        encoded_seqs, _ = to_dlomix(input_df)

        return self.model.predict(encoded_seqs)[:, int(self.mode) - 1]


model_wrappers = {
    "prosit": PrositIntensityWrapper,
    "transformer": TransformerIntensityWrapper,
    "torch": TorchIntensityWrapper,
    "koina": KoinaWrapper,
    "charge": ChargeStateWrapper,
}

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

import subprocess
import requests
import json
from copy import deepcopy

def get_installed_packages():
    packages = []
    key_packages = {'torch', 'dlomix', 'koinapy'}
    try:
        # Run the conda list command
        result = subprocess.run(['conda', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command was successful and get installed key packages
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                for key_package in key_packages:
                    if line.startswith(key_package):
                        print(line)
                        packages.append(key_package)
            return packages
        else:
            print("Error:", result.stderr)
            return packages
    except Exception as e:
        print("An exception occurred:", e)
        return packages


installed_packages = get_installed_packages()

if 'torch' in installed_packages:
    from . import PeptideEncoderModel
    from . import PrositModel
    import torch as th

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

if 'koinapy' in installed_packages:
    from koinapy import Koina

if 'dlomix' in installed_packages:
    # from . import TransformerModel
    from .ChargeState import custom_keras_utils as cutils
    from .ChargeState.chargestate import ChargeStateDistributionPredictor
    from .ChargeState.dlomix_preprocessing import to_dlomix
    import keras
    import tensorflow as tf

amino_acid_list = list("AVILMFYWSTNQCUGPRHKDEU")
def contains_amino_acid(token):
    
    # remove modiffication, if necessary
    token_ = re.sub(r"\[UNIMOD:[0-9]{1,3}]", '', token)
    return True if token_ in amino_acid_list else False

def hx(tokens):
    sequence = tokens[:, :-3]
    collision_energy = tf.strings.to_number(tokens[:, -2:-1])
    precursor_charge = tf.one_hot(
        tf.cast(tf.strings.to_number(tokens[:, -3:-2]), tf.int32) - 1, 6
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

class TorchIntensityWrapper(ModelWrapper):
    def __init__(self,
                 ion_dict_path: Union[str, bytes, os.PathLike],
                 token_dict_path: Union[str, bytes, os.PathLike],
                 yaml_dir_path: Union[str, bytes, os.PathLike],
                 mode: str,
                 ) -> None:
        ion_dict = pd.read_csv(ion_dict_path, index_col='full')
        self.ion_dict = ion_dict
        self.ion_ind = ion_dict.loc[mode]['index']
        with open(os.path.join(yaml_dir_path, "model.yaml")) as f:
            self.model_config = yaml.safe_load(f)
        with open(os.path.join(yaml_dir_path, "loader.yaml")) as f:
            load_config = yaml.safe_load(f)
        self.num_tokens = len(open(token_dict_path).read().strip().split("\n")) + 1
        self.token_dict = self.create_dictionary(token_dict_path)
        self.max_charge = load_config['charge'][1]
        # Same code as in loader_hf.py
        self.method_dic = {method: m for m, method in enumerate(load_config['method_list'])}
        self.method_dicr = {n: m for m, n in self.method_dic.items()}

        """
        self.model = PeptideEncoder(
            tokens = self.num_tokens,
            final_units = len(self.ion_dict),
            max_charge = load_config['charge'][-1],
            **model_config
        )
        self.model.to(device)
        self.model.load_state_dict(th.load(model_path, map_location=device))
        self.model.eval()
        """

    def create_dictionary(self, dictionary_path):
        amod_dic = {
            line.split()[0]: m for m, line in enumerate(open(dictionary_path))
        }
        amod_dic['X'] = len(amod_dic)
        amod_dic[''] = amod_dic['X']
        return amod_dic

    def hx(self, linear_input):
        """
        Turn list of [seq, energy, charge, method] into model ready tensors
        """
        sequence = linear_input[:, :-3]
        intseq = th.tensor(
            [
                integerize_sequence(seq, self.token_dict)
                for seq in sequence
            ],
            dtype=th.int32, device=device
        )
        charge = th.tensor(linear_input[:, -3].astype(int), dtype=th.int32, device=device)
        energy = th.tensor(linear_input[:, -2].astype(float), dtype=th.float32, device=device)
        method = th.tensor(
            [self.method_dic[tok] for tok in linear_input[:, -1]],
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
        prediction = (out[:, self.ion_ind.values]).detach().cpu().numpy()
        return prediction


class TorchPE(TorchIntensityWrapper):
    def __init__(self,
                 model_path: Union[str, bytes, os.PathLike],
                 ion_dict_path: Union[str, bytes, os.PathLike],
                 token_dict_path: Union[str, bytes, os.PathLike],
                 yaml_dir_path: Union[str, bytes, os.PathLike],
                 mode: str,
                 **kwargs
                 ) -> None:
        super().__init__(
            ion_dict_path=ion_dict_path,
            token_dict_path=token_dict_path,
            yaml_dir_path=yaml_dir_path,
            mode=mode,
        )

        self.model = PeptideEncoderModel(
            tokens=self.num_tokens,
            final_units=len(self.ion_dict),
            max_charge=self.max_charge,
            kwargs=self.model_config
        )
        self.model.to(device)
        self.model.load_state_dict(th.load(model_path, map_location=device))
        self.model.eval()


class TorchProsit(TorchIntensityWrapper):
    def __init__(self,
                 model_path: Union[str, bytes, os.PathLike],
                 ion_dict_path: Union[str, bytes, os.PathLike],
                 token_dict_path: Union[str, bytes, os.PathLike],
                 yaml_dir_path: Union[str, bytes, os.PathLike],
                 mode: str,
                 **kwargs
                 ) -> None:
        super().__init__(
            ion_dict_path=ion_dict_path,
            token_dict_path=token_dict_path,
            yaml_dir_path=yaml_dir_path,
            mode=mode,
        )

        self.model = PrositModel(
            tokens=self.num_tokens,
            final_units=len(self.ion_dict),
            max_charge=self.max_charge,
            kwargs=self.model_config
        )
        self.model.to(device)
        self.model.load_state_dict(th.load(model_path, map_location=device))
        self.model.eval()


class KoinaWrapper(ModelWrapper):
    def __init__(
            self,
            model_path: Union[str, bytes, os.PathLike],
            mode: str,
            server_url: Union[str, None]="koina.wilhelmlab.org:443",
            ssl: bool=True,
            inputs_ignored: int = 3,
            **kwargs
    ) -> None:
        
        server_url = "koina.wilhelmlab.org:443" if server_url == None else server_url
        self.model = Koina(model_path, server_url=server_url, ssl=ssl)
        self.mode = mode
        self.inputs_ignored = inputs_ignored

    def make_prediction(self, inputs: ndarray, silent: bool = True) -> ndarray:
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
        results = {}
        missing_values = {}
        for mode in self.mode:
            results[mode] = []
            missing_values[mode] = 0

            ann_bool = preds["annotation"] == bytes(mode, "utf-8")

            # If you don't find it, return 0 prediction
            if len(preds[ann_bool]["intensities"]) < len(sequences):
                if silent == False: print(preds)
                for i in range(len(sequences)):
                    if i not in preds[ann_bool]["intensities"].index:
                        results[mode].append(0.0)
                        missing_values[mode] += 1
                    else:
                        results[mode].append(preds[ann_bool]["intensities"][i])
            else:
                results[mode] = preds[ann_bool]["intensities"].to_list()

            assert len(results[mode]) == len(sequences)

        return pd.DataFrame(results).to_numpy()

class KoinaAC(KoinaWrapper):
    def __init__(
            self,
            model_path: Union[str, bytes, os.PathLike],
            mode: str,
            server_url: Union[str, None]="koina.wilhelmlab.org:443",
            ssl: bool=True,
            inputs_ignored: int = 3,
            **kwargs
    ) -> None:
        super().__init__(
            model_path=model_path,
            mode=mode,
            server_url=server_url,
            ssl=ssl,
            inputs_ignored=inputs_ignored,
            **kwargs
        )
    
    def make_prediction(self, inputs: ndarray, silent: bool = True) -> ndarray:
        
        bs, sl = inputs.shape
        
        # MUST make copy
        # - If you change [] to '', it ruins the mask_pep blanks process
        COPY = deepcopy(inputs).astype('U23')

        sequences = []
        # Add dashes to beginning and end
        COPY[COPY=='[]'] = ''
        for m in range(bs):
            if contains_amino_acid(COPY[m,0]) == False:
                COPY[m,0] += '-'
        for i in COPY[:, :-self.inputs_ignored]:
            try:
                sequence = "".join(i)
            except:
                sequence = b"".join(i).decode("utf-8")
            sequences.append(sequence)
        # FIXME This always assumes charge is third to last and collision energy second
        # to last (method is last). Need a dynamic way of finding these two features'
        # positions.
        input_dict = {
            "peptide_sequences": np.array(sequences),
            "precursor_charges": inputs[:, -3].astype("int"),
            "collision_energies": (inputs[:, -2].astype("float")),
            "fragmentation_types": np.array(list(map(lambda x: {'CID':1,'HCD':2}[x], inputs[:,-1]))).astype("int"),
        }
        counter = 0
        success = False
        while counter < 5 and not success:
            try:
                preds = self.model.predict(
                    pd.DataFrame(input_dict), 
                    min_intensity=-0.00001,
                    disable_progress_bar=True,
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
        results = {}
        missing_values = {}
        for mode in self.mode:
            results[mode] = []
            missing_values[mode] = 0

            ann_bool = preds["annotation"] == bytes(mode, "utf-8")

            # If you don't find it, return 0 prediction
            if len(preds[ann_bool]["intensities"]) < len(sequences):
                if silent == False: print(preds)
                for i in range(len(sequences)):
                    if i not in preds[ann_bool]["intensities"].index:
                        results[mode].append(0.0)
                        missing_values[mode] += 1
                    else:
                        results[mode].append(preds[ann_bool]["intensities"][i])
            else:
                results[mode] = preds[ann_bool]["intensities"].to_list()

            assert len(results[mode]) == len(sequences)

        return pd.DataFrame(results).to_numpy()

class ChargeStateWrapper(ModelWrapper):
    def __init__(
            self,
            model_path: Union[str, bytes, os.PathLike],
            mode: str,
            **kwargs
    ) -> None:

        self.path = "./src/models/ChargeState/trained_model.keras"
        if model_path:
            warnings.warn(
                """
                            You have given a path even though this project provides a model already.
                            If provided a .keras model in the same form as the provided model,
                            the code might still work. Otherwise, stick with provided model or modify this code section.
                            """
            )
            self.path = model_path
        self.mode = []
        for charge in mode:
            self.mode.append(int(charge[-1]) - 1)

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
        #print("PRINTING HX INPUTS:")
        #print(hx(inputs)["sequence"][0])

        sequences = []
        for seq in hx(inputs)["sequence"]:
            try:
                sequence = "".join(seq)
            except:
                sequence = b"".join(seq).decode("utf-8")
            sequences.append(sequence)

        input_df = pd.DataFrame()
        input_df["peptide_sequences"] = np.array(sequences)

        #for s in input_df["peptide_sequences"]:
        #    print(s)

        encoded_seqs, _ = to_dlomix(input_df)

        #print(self.model.predict(encoded_seqs)[:, self.mode].shape)
        return self.model.predict(encoded_seqs)[:, self.mode]

class FlyabilityWrapper(ModelWrapper):

    def __init__(
            self,
            model_path: Union[str, bytes, os.PathLike],
            mode: str,
            **kwargs
    ) -> None:

        if model_path:
            warnings.warn(
                """
                            You have given a path even though this project provides a model already.
                            No custom flyabilty models supported yet, using jesse's model...
                """
            )

        self.mode = []
        for fly in mode:
            self.mode.append(int(fly[-1]) - 1)
        #self.batch_num = 0

    def make_prediction(self, inputs: ndarray) -> ndarray:

        sequences = []
        for seq in hx(inputs)["sequence"]:
            try:
                sequence = "".join(seq)
            except:
                sequence = b"".join(seq).decode("utf-8")
            sequences.append(sequence)

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }

        data = f'''
        {{
            "id": "test",
            "inputs": [
                {{
                    "name": "peptide_sequences",
                    "datatype": "BYTES",
                    "shape": {[len(sequences), 1]},
                    "data": {json.dumps(sequences)}           
                }}
            ]
        }}
        '''
        #print(data)
        response = requests.post('http://10.157.98.63:8501/v2/models/pfly_2024/infer', headers=headers, data=data)
        pred_dict = json.loads(response.text)
        pred = np.array(pred_dict["outputs"][0]["data"]).reshape(pred_dict["outputs"][0]["shape"],)

        #if self.batch_num % 100 == 0:
        #    print(f"Processing Batch {self.batch_num}")
        #self.batch_num += 1

        return pred[:, self.mode]


class KoinaCCS(KoinaWrapper):
    def make_prediction(self, inputs: ndarray, silent: bool = True) -> ndarray:
        sequences = []
        for seq in hx(inputs)["sequence"]:
            try:
                sequence = "".join(seq)
            except:
                sequence = b"".join(seq).decode("utf-8")
            sequences.append(sequence)

        input_df = pd.DataFrame()
        input_df["peptide_sequences"] = np.array(sequences)

        precursor_charges = []
        charge_helper = np.array([1, 2, 3, 4, 5, 6])
        for arr in hx(inputs)["precursor_charge"]:
            precursor_charges.append(np.sum(arr * charge_helper))
        input_df["precursor_charges"] = precursor_charges

        pred = (self.model.predict(input_df)["ccs"]).to_numpy()
        pred = np.expand_dims(pred, axis=1)
        return pred


class KoinaIRT(KoinaWrapper):
    def make_prediction(self, inputs: ndarray, silent: bool = True) -> ndarray:
        sequences = []
        for seq in hx(inputs)["sequence"]:
            try:
                sequence = "".join(seq)
            except:
                sequence = b"".join(seq).decode("utf-8")
            sequences.append(sequence)

        input_df = pd.DataFrame()
        input_df["peptide_sequences"] = np.array(sequences)

        pred = (self.model.predict(input_df)["irt"]).to_numpy()
        pred = np.expand_dims(pred, axis=1)
        return pred


model_wrappers = {
    "torch_pe": TorchPE,
    "torch_prosit": TorchProsit,
    "koina": KoinaWrapper,
    "koina_irt": KoinaIRT,
    "koina_ccs": KoinaCCS,
    "koina_ac": KoinaAC,
    "charge": ChargeStateWrapper,
    "flyability": FlyabilityWrapper,
}

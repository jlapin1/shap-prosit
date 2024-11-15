import os
from abc import ABC, abstractmethod
from time import sleep
from typing import Union

#from koinapy import Koina
import pandas as pd
import numpy as np
import torch as th
import yaml
from . import PeptideEncoder
from numpy import ndarray

device = th.device("cuda" if th.cuda.is_available() else "cpu")

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

def tokenize_sequence(seq, dictionary):
    return [dictionary[aa] for aa in seq]

class ModelWrapper(ABC):
    @abstractmethod
    def __init__(self, path: Union[str, bytes, os.PathLike], ion: str) -> None:
        """Initialize model wrapper by loading the model."""
        pass

    @abstractmethod
    def make_prediction(self, inputs: ndarray) -> ndarray:
        """Make prediction from input array containing
        sequences, precursor charges and collision energy."""
        pass

"""
class PrositIntensityWrapper(ModelWrapper):
    def __init__(self, path: Union[str, bytes, os.PathLike], ion: str) -> None:
        self.ion_ind = annotations()[ion]
        self.model = PrositIntensityPredictor(seq_length=30)
        latest_checkpoint = tf.train.latest_checkpoint(path)
        self.model.load_weights(latest_checkpoint)

    def make_prediction(self, inputs: ndarray) -> ndarray:
        return (self.model(hx(inputs), training=False)[:, self.ion_ind]).numpy()


class TransformerIntensityWrapper(ModelWrapper):
    def __init__(self, path: Union[str, bytes, os.PathLike], ion: str) -> None:
        self.ion_ind = annotations()[ion]
        with open(os.path.join(path, "model.yaml"), encoding="utf-8") as file:
            model_config = yaml.safe_load(file)
        self.model = TransformerModel(**model_config)
        latest_checkpoint = tf.train.latest_checkpoint(path)
        self.model.load_weights(latest_checkpoint)
    
    def make_prediction(self, inputs: ndarray) -> ndarray:
        return (self.model(hx(inputs), training=False)[:, self.ion_ind]).numpy()
"""

class TorchIntensityWrapper(ModelWrapper):
    def __init__(self, 
        model_path: Union[str, bytes, os.PathLike], 
        ion_dict_path: Union[str, bytes, os.PathLike],
        token_dict_path: Union[str, bytes, os.PathLike],
        yaml_dir_path: Union[str, bytes, os.PathLike],
        ion: str,
    ) -> None:
        ion_dict = pd.read_csv(ion_dict_path, index_col='full')
        self.ion_ind = ion_dict.loc[ion]['index']
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

    def create_dictionary(self, dictionary_path):
        amod_dic = {
            line.split()[0]:m for m, line in enumerate(open(dictionary_path))
        }
        amod_dic['X'] = len(amod_dic)
        amod_dic[''] = amod_dic['X']
        return amod_dic

    def hx(self, tokens):
        sequence = tokens[:,:-2]
        intseq = th.tensor(
            [tokenize_sequence(seq, self.token_dict) for seq in sequence], 
            dtype=th.int32, device=device
        )
        charge = th.tensor(tokens[:,-1].astype(int), dtype=th.int32, device=device)
        energy = th.tensor(tokens[:,-2].astype(float), dtype=th.float32, device=device)
        return {
            'intseq': intseq,
            'charge': charge,
            'energy': energy,
        }

    def make_prediction(self, inputs: ndarray) -> ndarray:
        with th.no_grad():
            out = self.model(**self.hx(inputs))
        out = out / out.max(dim=1, keepdim=True)[0]
        prediction = (out[:, self.ion_ind]).detach().cpu().numpy()
        return prediction

class KoinaWrapper(ModelWrapper):
    def __init__(self, path: Union[str, bytes, os.PathLike], ion: str) -> None:
        self.model = Koina(path)
        self.ion = ion

    def make_prediction(self, inputs: ndarray) -> ndarray:
        sequences = []
        for i in inputs[:, :-2]:
            try:
                seq = "".join(i)
            except:
                seq = b"".join(i).decode("utf-8")
            sequences.append(seq)
        input_dict = {
            "peptide_sequences": np.array(sequences),
            "precursor_charges": inputs[:, -1].astype("int"),
            "collision_energies": (inputs[:, -2].astype("float") * 100).astype("int"),
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
        if len(
            preds[preds["annotation"] == bytes(self.ion, "utf-8")]["intensities"]
        ) < len(sequences):
            print(preds)
            results = []
            for i in range(len(sequences)):
                if (
                    i
                    not in preds[preds["annotation"] == bytes(self.ion, "utf-8")][
                        "intensities"
                    ].index
                ):
                    results.append(0.0)
                else:
                    results.append(
                        preds[preds["annotation"] == bytes(self.ion, "utf-8")][
                            "intensities"
                        ][i]
                    )
        else:
            results = preds[preds["annotation"] == bytes(self.ion, "utf-8")][
                "intensities"
            ].to_numpy()
        return np.array(results)


model_wrappers = {
    #"prosit": PrositIntensityWrapper,
    #"transformer": TransformerIntensityWrapper,
    "torch": TorchIntensityWrapper,
    #"koina": KoinaWrapper,
}

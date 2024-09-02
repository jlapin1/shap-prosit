import os
from abc import ABC, abstractmethod
from typing import Union

from koinapy import Koina
import pandas as pd
import numpy as np
import tensorflow as tf
import yaml
from dlomix.models import PrositIntensityPredictor
from numpy import ndarray

from . import TransformerModel

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
            "precursor_charges": inputs[:, -2].astype("float"),
            "collision_energies": inputs[:, -1].astype("float"),
        }
        preds = self.model.predict(pd.DataFrame(input_dict), min_intensity=-0.00001)
        if len(preds[preds["annotation"] == bytes(self.ion, "utf-8")]["intensities"]) < len(sequences):
            print(sequences[0])
            results = []
            for i in range(len(sequences)):
                if i not in preds[preds["annotation"] == bytes(self.ion, "utf-8")]["intensities"].index:
                    results.append(0.0)
                else:
                    results.append(preds[preds["annotation"] == bytes(self.ion, "utf-8")]["intensities"][i])
        else:
            results = preds[preds["annotation"] == bytes(self.ion, "utf-8")]["intensities"].to_numpy()
        return np.array(results)


model_wrappers = {
    "prosit": PrositIntensityWrapper,
    "transformer": TransformerIntensityWrapper,
    "koina": KoinaWrapper,
}

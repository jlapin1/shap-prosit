import os
from abc import ABC, abstractmethod
from time import sleep
from typing import Union
import warnings

from koinapy import Koina
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import yaml
from dlomix.models import PrositIntensityPredictor
from numpy import ndarray

from . import TransformerModel
from .ChargeState import custom_keras_utils as cutils
from .ChargeState.chargestate import ChargeStateDistributionPredictor
from .ChargeState.dlomix_preprocessing import to_dlomix


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
        # set ion = "rt" to enter retention time mode
        # inputs["sequences"] = [["W", "E"],[],[]]
        if self.ion == "rt":

            print(type(hx(inputs)["sequence"][0]))

            sequences = []
            for seq in hx(inputs)["sequence"]:
                try:
                    sequence = "".join(seq)
                except:
                    sequence = b"".join(seq).decode("utf-8")
                sequences.append(sequence)

            input_df = pd.DataFrame()
            input_df['peptide_sequences'] = np.array(sequences)

            return (self.model.predict(input_df)["irt"]).to_numpy()

        elif self.ion == "cc":
            # set ion = "cc" to enter retention time mode
            sequences = []
            for seq in hx(inputs)["sequence"]:
                try:
                    sequence = "".join(seq)
                except:
                    sequence = b"".join(seq).decode("utf-8")
                sequences.append(sequence)

            input_df = pd.DataFrame()
            input_df['peptide_sequences'] = np.array(sequences)
            input_df['precursor_charges'] = inputs[:, -1].astype("int")
            print(input_df['peptide_sequences'])
            print(input_df['precursor_charges'])
            assert len(input_df.shape)==2, f"shape is not 2-dimensional\n{input_df}"
            return (self.model.predict(input_df)["ccs"]).to_numpy()

        else:
            # stuff for intensity model
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

class ChargeStateWrapper(ModelWrapper):
    def __init__(self, path: Union[str, bytes, os.PathLike], ion: str) -> None:
        import ChargeState
        self.path = "./src/models/ChargeState/trained_model.keras"
        if path:
            warnings.warn('''
                            You have given a path even though this project provides a model already. 
                            If provided a .keras model in the same form as the provided model, 
                            the code might still work. Otherwise, stick with provided model or modify this code section.
                            ''')
            self.path = path
        self.ion = ion[-1]

        self.model = keras.saving.load_model(
            self.path,
            custom_objects={
                'upscaled_mean_squared_error': cutils.upscaled_mean_squared_error,
                'euclidean_similarity': cutils.euclidean_similarity,
                'masked_pearson_correlation_distance': cutils.masked_pearson_correlation_distance,
                'masked_spectral_distance': cutils.masked_spectral_distance,
                'ChargeStateDistributionPredictor': ChargeStateDistributionPredictor
            }
)

    def make_prediction(self, inputs: ndarray) -> ndarray:

        # inputs["sequences"] = [["W", "E"],[],[]]

        print(type(hx(inputs)["sequence"][0]))

        sequences = []
        for seq in hx(inputs)["sequence"]:
            try:
                sequence = "".join(seq)
            except:
                sequence = b"".join(seq).decode("utf-8")
            sequences.append(sequence)

        input_df = pd.DataFrame()
        input_df['peptide_sequences'] = np.array(sequences)

        encoded_seqs, _ = to_dlomix(input_df)

        return self.model.predict(encoded_seqs)[:,int(self.ion)-1]


model_wrappers = {
    "prosit": PrositIntensityWrapper,
    "transformer": TransformerIntensityWrapper,
    "koina": KoinaWrapper,
    "charge": ChargeStateWrapper
}

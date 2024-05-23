import os
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import tensorflow as tf
import yaml
from dlomix.models import PrositIntensityPredictor
from numpy import ndarray

from . import TransformerModel


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
    def make_prediction(self, input: dict) -> ndarray:
        """Make prediction from input dictionary containing
        sequences, precursor charges and collision energy."""
        pass


class PrositIntensityWrapper(ModelWrapper):
    def __init__(self, path: Union[str, bytes, os.PathLike], ion: str) -> None:
        self.ion_ind = annotations()[ion]
        self.model = PrositIntensityPredictor(seq_length=30)
        latest_checkpoint = tf.train.latest_checkpoint(path)
        self.model.load_weights(latest_checkpoint)

    def make_prediction(self, input: dict) -> ndarray:
        return (self.model(input, training=False)[:, self.ion_ind]).numpy()


class TransformerIntensityWrapper(ModelWrapper):
    def __init__(self, path: Union[str, bytes, os.PathLike], ion: str) -> None:
        self.ion_ind = annotations()[ion]
        with open(os.path.join(path, "model.yaml"), encoding="utf-8") as file:
            model_config = yaml.safe_load(file)
        self.model = TransformerModel(**model_config)
        latest_checkpoint = tf.train.latest_checkpoint(path)
        self.model.load_weights(latest_checkpoint)

    def make_prediction(self, input: dict) -> ndarray:
        return (self.model(input, training=False)[:, self.ion_ind]).numpy()


model_wrappers = {
    "prosit": PrositIntensityWrapper,
    "transformer": TransformerIntensityWrapper,
}

"""Module for training of intensity model."""

import os
from typing import Union

from dlomix.data import IntensityDataset
from dlomix.losses import masked_spectral_distance
from dlomix.models import PrositIntensityPredictor


class IntensityModelTrainer:  # TODO: class is unnecessary, one method would be enought, intensity dataset must be provided
    """Class for training of intensity model."""

    def __init__(
        self,
        data_path: Union[str, bytes, os.PathLike],
        seq_len: int = 30,
        batch_size: int = 64,
    ):
        """Create dataset and initialize model.

        Args:
            data_path (Union[str, bytes, os.PathLike]): Path to intensity data.
            seq_len (int, optional): Length of sequences in data set. Defaults to 30.
            batch_size (int, optional): Size of batch in forward pass. Defaults to 64.
        """
        self.intdata = IntensityDataset(
            data_source=data_path,
            seq_length=seq_len,
            batch_size=batch_size,
            val_ratio=0.2,
            test=False,
        )

        print(f"Training examples: {batch_size * len(self.intdata.train_data)}")
        print(f"Validation examples: {batch_size * len(self.intdata.val_data)}")

        self.model = PrositIntensityPredictor(seq_length=seq_len)

    def train_model(self, epochs: int) -> PrositIntensityPredictor:
        """Train intensity predictor.

        Args:
            epochs (int): Number of epochs to train the model.

        Returns:
            PrositIntensityPredictor: Trained model.
        """
        self.model.compile(
            optimizer="adam",
            loss=masked_spectral_distance,
        )

        self.model.fit(
            self.intdata.train_data,
            validation_data=self.intdata.val_data,
            epochs=epochs,
        )

        return self.model


def main():
    """Main script to train and save the model."""

    trainer = IntensityModelTrainer(
        data_path="./intensity_data.csv", seq_len=30, batch_size=64
    )
    model = trainer.train_model(epochs=20)
    model.save_weights("saved_model/savedmodel")


if __name__ == "__main__":
    main()

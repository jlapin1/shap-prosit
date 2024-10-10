"""Module for training of intensity model."""

import os
from typing import Union
import tensorflow as tf

from dlomix.data import IntensityDataset
from dlomix.losses import masked_spectral_distance
from dlomix.models import PrositIntensityPredictor


class IntensityModelTrainer:
    """Class for training of intensity model."""

    def __init__(
        self,
        data_path: Union[str, bytes, os.PathLike],
        val_path: Union[str, bytes, os.PathLike],
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
            test=False,
        )

        self.valdata = IntensityDataset(
            data_source=val_path,
            seq_length=seq_len,
            batch_size=batch_size,
            test=False,
        )

        print(f"Training examples: {batch_size * len(self.intdata.train_data)}")
        print(f"Validation examples: {batch_size * len(self.valdata.train_data)}")

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

        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="saved_model/{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
            initial_value_threshold=None,
        )

        self.model.fit(
            self.intdata.train_data,
            validation_data=self.valdata.train_data,
            epochs=epochs,
            callbacks=callback,
        )

        return self.model


def main():
    """Main script to train and save the model. Also saves validation set."""

    trainer = IntensityModelTrainer(
        data_path="train_data.csv", val_path="val_data.csv", seq_len=30, batch_size=64
    )
    model = trainer.train_model(epochs=200)

    # Generate background dataset and save to the file.
    inps = []
    for i in trainer.valdata.val_data:
        charges = i[0]["precursor_charge"].numpy().argmax(1) + 1
        for j in range(i[0]["sequence"].shape[0]):
            csseq = ",".join([k.numpy().decode("utf-8") for k in i[0]["sequence"][j]])
            ce = i[0]["collision_energy"][j].numpy()[0]
            inp = csseq + ",%.2f,%d" % (ce, charges[j])
            inps.append(inp)
    with open("val_inps.csv", "w") as f:
        f.write("\n".join(inps))


if __name__ == "__main__":
    main()

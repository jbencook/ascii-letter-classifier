from typing import Union

import tensorflow as tf

from mnist import load_dataset

from .model import ascii_letter_classifier
from .config import AsciiLetterConfig
from .files import AsciiLetterFiles


def train_model(config: Union[str, AsciiLetterConfig] = AsciiLetterConfig()) -> None:
    """Train the model and save classifier and feature weights."""
    if isinstance(config, str):
        config = AsciiLetterConfig.from_yaml(config)
    files = AsciiLetterFiles(config)
    x_train, y_train = load_dataset(files.train_dataset, config)
    x_test, y_test = load_dataset(files.test_dataset, config)
    model = ascii_letter_classifier(config)
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        verbose=config.verbose,
        epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_steps=config.validation_steps,
    )
    model.save_weights(files.model_weights, overwrite=True)
    return 'OK'

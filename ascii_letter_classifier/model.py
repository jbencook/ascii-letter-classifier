from typing import Union

import tensorflow as tf
from mnist import mnist_classifier

from .config import AsciiLetterConfig
from .files import AsciiLetterFiles


def ascii_letter_classifier(
        config: Union[str, AsciiLetterConfig] = AsciiLetterConfig()
) -> tf.keras.Model:
    """MNIST classifier with different input resolution and number of classes."""
    if isinstance(config, str):
        config = AsciiLetterConfig.from_yaml(config)
    model = mnist_classifier(config.update(pretrained_classifier=False))
    if config.pretrained_classifier:
        files = AsciiLetterFiles(config)
        model.load_weights(files.model_weights)
    return model

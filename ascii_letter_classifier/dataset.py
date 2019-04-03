import os
import string
import random
from io import BytesIO
from typing import Union, List, Dict, Tuple
from pathlib import Path

import imageio
import tensorflow as tf
from mnist import write_holdout

from .config import AsciiLetterConfig
from .files import AsciiLetterFiles


def save_datasets(config: Union[str, AsciiLetterConfig] = AsciiLetterConfig()) -> None:
    """Save train and test TFRecord files."""
    if isinstance(config, str):
        config = AsciiLetterConfig.from_yaml(config)
    files = AsciiLetterFiles(config)
    files.download_tarball()
    files.extract_dataset_folder()

    # Create list of samples
    samples = []
    dataset_folder = Path(files.dataset_folder)
    for label in os.listdir(dataset_folder):
        class_folder = dataset_folder/label
        for fname in os.listdir(class_folder):
            img = imageio.imread(class_folder/fname)
            class_index = string.ascii_lowercase.index(label)
            samples.append(dict(
                img=img,
                label=class_index,
            ))

    # Train/test split
    random.seed(config.seed)
    random.shuffle(samples)
    train_samples = samples[:config.n_train_samples]
    test_samples = samples[-config.n_test_samples:]
    write_holdout(train_samples, files.train_dataset)
    write_holdout(test_samples, files.test_dataset)

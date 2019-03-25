import os
import string
import random
from io import BytesIO
from typing import Union, List, Dict
from pathlib import Path

import imageio
import tensorflow as tf

from .config import AsciiLetterConfig
from .files import AsciiLetterFiles


def _int64_feature(value: int) -> tf.train.Feature:
    """int64 feature wrapper"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """bytes feature wrapper"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_holdout(samples: List[Dict], file_path: str) -> None:
    """Write a single TFRecord file for either train or test."""
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for sample in samples:
            buffer = BytesIO()
            img = sample['img']
            imageio.imwrite(buffer, img, format='jpg')
            buffer.seek(0)
            feature = dict(
                image=_bytes_feature(buffer.read()),
                label=_int64_feature(int(sample['label'])),
            )
            example = tf.train.Example(
                features=tf.train.Features(feature=feature),
            )
            writer.write(example.SerializeToString())


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
    _write_holdout(train_samples, files.train_dataset)
    _write_holdout(test_samples, files.test_dataset)

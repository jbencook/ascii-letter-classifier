from pathlib import Path
from typing import Tuple

from dataclasses import dataclass
from mnist import MnistConfig


@dataclass
class AsciiLetterConfig(MnistConfig):
    n_classes: int = 26
    artifact_directory: str = str(Path.home()/'.mlpipes/ascii_letter')
    seed: int = 12345
    n_train_samples: int = 2080
    n_test_samples: int = 520
    image_height: int = 32
    image_width: int = 32
    batch_size: int = 128
    n_epochs: int = 1
    learning_rate: float = 0.005
    pretrained_features: bool = True

    def update(self, **kwargs) -> 'AsciiLetterConfig':
        params = {**self.__dict__, **kwargs}
        return AsciiLetterConfig(**params)

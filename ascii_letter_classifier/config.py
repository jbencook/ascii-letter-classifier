from pathlib import Path

from dataclasses import dataclass

from mnist.config import MnistConfig


@dataclass
class AsciiLetterConfig(MnistConfig):
    n_classes: int = 26
    artifact_directory: str = str(Path.home()/'.mlpipes/ascii_letter')

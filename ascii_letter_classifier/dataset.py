from typing import Union

from .config import AsciiLetterConfig
from .files import AsciiLetterFiles


def save_datasets(config: Union[str, AsciiLetterConfig] = AsciiLetterConfig()) -> None:
    """Save train and test TFRecord files."""
    if isinstance(config, str):
        config = AsciiLetterConfig.from_yaml(config)
    files = AsciiLetterFiles(config)
    files.download_tarball()
    files.extract_dataset_folder()

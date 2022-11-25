import os
import tarfile
from typing import Union
from pathlib import Path
from urllib import request

from .config import AsciiLetterConfig
from .__version__ import __version__


class AsciiLetterFiles:
    _artifact_url_prefix: str = 'https://s3.amazonaws.com/mlpipes/ascii_letter/'
    _dataset_tarball_filename: str = 'images.tar.gz'
    _dataset_folder: str = 'images'
    _train_dataset_filename: str = f"train-{__version__['dataset']}.tfrecord"
    _test_dataset_filename: str = f"test-{__version__['dataset']}.tfrecord"
    _feature_weights_filename: str = f'mnist-features-0.2.1.h5'
    _model_weights_filename: str = f"ascii-letter-classifier-{__version__['model']}.h5"

    def __init__(self, config: Union[str, AsciiLetterConfig] = AsciiLetterConfig()) -> None:
        if isinstance(config, str):
            config = AsciiLetterConfig.from_yaml(config)
        self._directory = config.artifact_directory_path
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_tarball(self) -> str:
        return str(self._directory/self._dataset_tarball_filename)

    def download_tarball(self) -> str:
        if not Path(self.dataset_tarball).exists():
            url = os.path.join(self._artifact_url_prefix, self._dataset_tarball_filename)
            request.urlretrieve(url, self.dataset_tarball)
        return self.dataset_tarball

    @property
    def dataset_folder(self) -> str:
        return str(self._directory/self._dataset_folder)

    def extract_dataset_folder(self) -> str:
        if not Path(self.dataset_folder).exists():
            with tarfile.open(self.dataset_tarball) as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, self._directory)
        return self.dataset_folder

    @property
    def train_dataset(self) -> str:
        return str(self._directory/self._train_dataset_filename)

    @property
    def test_dataset(self) -> str:
        return str(self._directory/self._test_dataset_filename)

    @property
    def model_weights(self) -> str:
        return str(self._directory/self._model_weights_filename)

    def download_model_weights(self) -> str:
        if not Path(self.model_weights).exists():
            url = os.path.join(self._artifact_url_prefix, self._model_weights_filename)
            request.urlretrieve(url, self.model_weights)
        return self.model_weights

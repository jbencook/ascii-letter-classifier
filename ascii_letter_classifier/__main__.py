import fire

from .dataset import save_datasets


def main():
    fire.Fire({
        'save-datasets': save_datasets,
    })

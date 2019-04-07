import os
from pathlib import Path
from setuptools import setup

about = {}
path = Path('.')/'ascii_letter_classifier/__version__.py'
with open(path) as version:
    exec(version.read(), about)


setup(
    name='ascii-letter-classifier',
    version=about['__version__']['package'],
    packages=['ascii_letter_classifier'],
    license='MIT',
    install_requires=[
        'dataclasses',
        'fire',
        'imageio',
        'mnist-pipeline==0.2.1',
    ],
    extras_require={
        'cpu': ['tensorflow'],
        'gpu': ['tensorflow-gpu'],
    },
    entry_points={
        'console_scripts': [
            'letters = ascii_letter_classifier.__main__:main',
        ]
    }
)

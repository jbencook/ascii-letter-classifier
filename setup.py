from setuptools import setup

setup(
    name='ascii-letter-classifier',
    version='0.0.0',
    packages=['ascii_letter_classifier'],
    license='MIT',
    install_requires=[
        'dataclasses',
        'fire',
        'imageio',
        'mnist-pipeline>=0.2.0',
        'pyyaml',
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

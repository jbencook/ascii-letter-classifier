# ASCII Letter Classifier Pipeline
A toy ML pipeline that creates an ASCII letter classifier.

## Installation

``` shell
pip install ascii-letter-classifier[cpu]
```
...or swap in `[gpu]` at the end to use the `tensorflow-gpu` package.

## Inference quick start

Set `pretrained_classifier` to true in config to get a pretrained classifier. Then pass in a batch of TensorFlow or numpy images with shape `(batch_size, 32, 32, 1)` to `predict()`. If the images are 8-bit integers, divide by 255 to map to floats in `[0, 1]`.

``` python
import numpy as np
from ascii_letter_classifier import AsciiLetterConfig, ascii_letter_classifier

x = np.random.randint(
    0, 256,
    size=(1, 32, 32, 1),
    dtype=np.uint8
) / 255

model = ascii_letter_classifier(AsciiLetterConfig(pretrained_classifier=True))

y = model.predict(x).argmax()
```

## Running the pipeline

There are two commands in the pipeline:

1. `save-datasets`
2. `train-model`

Each command can be run with with `letters` CLI, e.g. `letters save-datasets`. Optionally, they can also both take the path to a YAML config file with overrides. The values in these files override the defaults set in [letters/config.py](./letters/config.py). So for example, to run the pipeline for 5 epochs, you can call `letters train-model config/pipeline.yml`.

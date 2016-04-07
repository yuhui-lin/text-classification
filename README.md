# Deep Learning Models for Text Classification
[![TeamCity CodeBetter](https://img.shields.io/teamcity/codebetter/bt428.svg)](yuhui-lin.github.io)
This repository contains several Deep learning models for text classification implemented in TensorFlow.

## Reqirement
* Python 2.7
* Numpy
* TensorFlow 0.71

## Running
```bash
python -m cnn_character.train --help
python -m cnn_character.train --dataset ag --datasets_dir ~/Downloads/text-classification/
```
## Components
* ``data/``: default directory for storing datasets.
* ``data_helper.py``: load datasets and preprocessing.

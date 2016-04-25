# Deep Learning Models for Text Classification
[![TeamCity CodeBetter](https://img.shields.io/teamcity/codebetter/bt428.svg)](yuhui-lin.github.io)

This repository contains several Deep learning models for text classification implemented in TensorFlow.

## Reqirement
* Python 2.7
* Numpy
* TensorFlow r0.8

## Running
```bash
# convert data to TFRecords
python -m convert_data --help
python -m convert_data --dataset rotten --datasets_dir ~/Downloads/text-classification/

# training
python -m cnn_character.train --help
python -m cnn_character.train --dataset rotten --datasets_dir ~/Downloads/text-classification/

# evaluation
python -m cnn_character.eval --help
<!--python -m cnn_character.eval --train_dir cnn_character/outputs/rotten.time/-->
python -m cnn_word.eval --checkpoint_dir cnn_word/logs/rotten.2016-04-24.17-00-27
```
## Components
* ``datasets/``: default directory for storing datasets and TFRecords.
* ``convert_data.py``: downloads datasets and converts original datasets to TFRecords file. It would be best to have all data preprocessing in this program , so we'll spend less time on training.
* ``input.py``: reads TFRecords files, shuffle and batch.
* ``model-name/train.py``: model training.
* ``model-name/model.py``: builds certain model.
* ``model-name/eval.py`` : evaluates models on cpu.

# Dataset
If dataset is not found under datasets_dir, it will be downloaded automatically. Currently we only use datasets that can be loaded entirely in memory. The feeding method is used now to get data into TF model.
* ``rotten``:
* ``ag``:
* ``newsgroups``:
* ``imdb``:



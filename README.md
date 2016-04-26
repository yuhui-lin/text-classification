# Deep Learning Models for Text Classification
[![TeamCity CodeBetter](https://img.shields.io/teamcity/codebetter/bt428.svg)](yuhui-lin.github.io)

This repository contains several Deep learning models for text classification implemented in TensorFlow.

## Reqirement
* Python 2.7
* Numpy
* TensorFlow r0.8

## Running
1. convert original dataset into TFRecords File.
2. start training the model from the TFRecords.
3. run eval.py to get evaluation on test set.

```bash
# convert data to TFRecords
python -m convert_data --help
python -m convert_data --dataset rotten --datasets_dir ~/Downloads/text-classification/
python -m convert_data --model_type embedding --dataset rotten --datasets_dir ~/Downloads/text-classification/

# training
python -m cnn_character.train --help
python -m cnn_character.train --dataset rotten --datasets_dir ~/Downloads/text-classification/
python -m cnn_character.train --dataset rotten --print_step 5 --summary_step 30 --checkpoint_step 300 --num_epochs 200
python -m rcnn_embedding.train --dataset rotten --print_step 5 --summary_step 30 --checkpoint_step 300 --num_epochs 200 --datasets_dir ~/Downloads/text-classification/


# evaluation
python -m cnn_character.eval --help
python -m cnn_character.eval --train_dir cnn_character/outputs/rotten.time/
```
## Components
* ``datasets/``: default directory for storing datasets and TFRecords.
* ``convert_data.py``: downloads datasets and converts original datasets to TFRecords file. It would be best to have all data preprocessing in this program , so we'll spend less time on training.
* ``input.py``: reads TFRecords files, shuffle and batch.
* ``model-name/train.py``: model training.
* ``model-name/model.py``: builds certain model.
* ``model-name/eval.py`` : evaluates models on cpu.

## Models
- ``cnn_character``: 9-layer large convolutional neural network based on raw character.
- ``rcnn_embedding``: Bi-reccurrent neural network with convolution based on pre-trained word vector.
- ``cnn_embedding``: relatively small model suitable for small datasets.

## Dataset
If dataset is not found under datasets_dir, it will be downloaded automatically. Currently we only use datasets that can be loaded entirely in memory. The feeding method is used now to get data into TF model.
* ``rotten``:This is a dataset for polarised review classification. They\footnote{\url{http://www.cs.cornell.edu/people/pabo/movie-review-data/}} provide a set of 5300 positive reviews and a set of 5300 negative reviews. We manually split eighty percent of them to be the training data and the remaining 20 percent to be test data. 
* ``ag``: AG\footnote{\url{http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html}} is a collection of more than 1 million news articles. News articles have been gathered from more than 2000  news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. 
* ``newsgroups``: Newsgroups\footnote{\url{http://qwone.com/~jason/20Newsgroups/}} data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of my knowledge, it was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews paper, though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.
* ``imdb``: This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. They\footnote{\url{http://ai.stanford.edu/~amaas/data/sentiment/}} provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided.



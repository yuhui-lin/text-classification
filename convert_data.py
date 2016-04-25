"""Converts different text classification datasets to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import urllib
import xml.etree.ElementTree as ET
import random
import bz2
import collections
import os
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "dataset", "rotten",
    'dataset used to train the neural network: rotten, ag, newsgroups, imdb. (default: rotten_tomato)')
tf.app.flags.DEFINE_string('datasets_dir', 'datasets',
                           'Directory to download data files and write the '
                           'converted result')
tf.app.flags.DEFINE_string(
    "model_type", "character",
    'generate different TFRecords for different models: character, embedding')
tf.app.flags.DEFINE_float(
    'test_ratio', 0.2,
    'the ratio of test data to separate from the original dataset.')
tf.app.flags.DEFINE_integer(
    "input_length", 1014,
    "number of characters in each input sequences (default: 1024)")
tf.app.flags.DEFINE_integer(
    "embed_length", 100,
    "number of characters in each input sequences (default: 1024)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "Dropout keep probability (default: 0.5)")

# datasets variables
# ==================================================
ROTTEN_SOURCE = "rt-polaritydata"
ROTTEN_DOWNLOADED = "rt-polaritydata.tar.gz"
ROTTEN_URL = "http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
AG_SOURCE = "newsspace200.xml"
AG_DOWNLOADED = "newsspace200.xml.bz"
AG_URL = "http://www.di.unipi.it/~gulli/newsspace200.xml.bz"
NEWSGROUPS_SOURCE = "20news-bydate"
NEWSGROUPS_DOWNLOADED = "20news-bydate.tar.gz"
NEWSGROUPS_URL = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
IMDB_SOURCE = "aclImdb"
IMDB_DOWNLOADED = "aclImdb_v1.tar.gz"
IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# only categories in this list will be considered
AG_CATEGORIES = ["World", "Sports", "Entertainment", "Business"]
# number of items for each category
AG_CATEGORIES_NUM = 40000

# AG's news info
# [('The recent onslaught of hurricanes has prompted some media outlets to mention  quot;global warming quot; as a possible cause, but a team of cli
# mate researchers set the record straight.', 1), ('ProEnglish, a national organization that wants to make English the official language of US gover
# nment operations, is suing the Department of Health and Human Services over a ', 1), ('5', 106), ('Music Feeds', 1207), ('Toons', 2150), ('Softwar
# e and Developement', 2739), (None, 11904), ('U.S.', 13770), ('Italia', 13814), ('Health', 19915), ('Europe', 30905), ('Top News', 31917), ('Sci/Te
# ch', 41194), ('Top Stories', 56045), ('Business', 56656), ('Sports', 62163), ('Entertainment', 70892), ('World', 81456)]
# ('length:', 18)

NEWSGROUPS_DICT = {"comp.graphics": "comp",
                   "comp.os.ms-windows.misc": "comp",
                   "comp.sys.ibm.pc.hardware": "comp",
                   "comp.sys.mac.hardware": "comp",
                   "comp.windows.x": "comp",
                   "talk.politics.misc": "politics",
                   "talk.politics.guns": "politics",
                   "talk.politics.mideast": "politics",
                   "rec.autos": "rec",
                   "rec.motorcycles": "rec",
                   "rec.sport.baseball": "rec",
                   "rec.sport.hockey": "rec",
                   "talk.religion.misc": "religion",
                   "alt.atheism": "religion",
                   "soc.religion.christian": "religion"}
NEWSGROUPS_CATEGORIES = list(set(NEWSGROUPS_DICT.itervalues()))

IMDB_CATEGORIES = ["neg", "pos"]

# model variables
# ==================================================
alphabet = r"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
test_ratio = 0.20
MIN_SEQU_LENGTH = 20


def maybe_download(data_dir, source_name, source_downloaded, source_url):
    """download the input data if not exists"""
    if not tf.gfile.Exists(data_dir):
        tf.gfile.MakeDirs(data_dir)
    source_path = os.path.join(data_dir, source_name)
    print("source path:", source_path)
    if not tf.gfile.Exists(source_path):
        download_path = os.path.join(data_dir, source_downloaded)
        print("downloading", download_path, "...")
        download_path, _ = urllib.urlretrieve(source_url, download_path)
        with tf.gfile.GFile(download_path) as p:
            size = p.Size
        print('Successfully downloaded', download_path, size, 'bytes.')
        print("extracting", download_path, "...")
        if download_path.endswith(".tar.gz"):
            with tarfile.open(download_path, "r:*") as f:
                f.extractall(data_dir)
            print("successfully extracted file")
        elif (download_path.endswith(".bz")):
            bzfile = bz2.BZ2File(download_path)
            data = bzfile.read()
            with open(source_path, "w") as new_source:
                new_source.write(data)
            print("successfully extracted file")
        else:
            print("unknown compressed file")

    print("dataset already exists:", source_path)
    return source_path


def create_one_hot_vector(cat, categories):
    """if cat is not found, all zero vector will be returned.
    if you call this to create labels, please make sure cat is in categories !!!
    cat is in categories. Create one-hot vector for cat.
    argv:
    cat: str or char
    categories: [str] or str
    return:
    one-hoe vector
    """
    ret = [0] * len(categories)
    # if cat is not found, all zero vector will be returned.
    try:
        i = categories.index(cat)
        if i >= 0:
            ret[i] = 1
    except ValueError:
        pass
    return ret


def shuffle_data(x, y, seed=-1):
    """shuffle sequences and its labels by seed in place."""
    print("shuffling data...")
    if seed < 0:
        seed = random.random()
    random.shuffle(x, lambda: seed)
    random.shuffle(y, lambda: seed)

    return [x, y]


def shuffle_and_split(x, y, test_ratio=0.2, seed=-1):
    print("shuffling and splitting data...")
    shuffle_data(x, y, seed)
    delimiter = int(len(y) * test_ratio * -1)
    x_train, x_test = x[:delimiter], x[delimiter:]
    y_train, y_test = y[:delimiter], y[delimiter:]

    return [x_train, y_train, x_test, y_test]


def grab_data_from_folder(source_path,
                          selected_folders,
                          str_filter=lambda x: x,
                          folder_map=lambda x: x):
    """grab data from datasets (20newsgroups, IMDB).
    each sequence is independent file stored under its category folder.
    argv:
        source_path: str
        selected_folders: [str]
        str_filter: function: remove redundant word in a sequence
        folder_map: map folder name to category name, return None if no corresponding cat.
    return:
        [sequences, labels]
    """
    sequences = []
    labels = []

    for folder in os.listdir(source_path):
        folder_path = os.path.join(source_path, folder)
        folder_cat = folder_map(folder)
        if os.path.isdir(
                folder_path) and folder_cat is not None and folder_cat in selected_folders:
            for fname in os.listdir(folder_path):
                file_path = os.path.join(folder_path, fname)
                if os.path.isfile(file_path):
                    with open(file_path) as f:
                        sequence = f.read()
                        if len(sequence) > MIN_SEQU_LENGTH:
                            sequence = str_filter(sequence)
                            sequences.append(sequence)
                            labels.append(create_one_hot_vector(
                                folder_cat, selected_folders))

    return [sequences, labels]


def grab_data_ag(source_path):
    sequences = []
    labels = []
    # number of sequences for each category
    count = dict(zip(AG_CATEGORIES, [0] * len(AG_CATEGORIES)))
    with open(source_path) as f:
        lines = f.readlines()
        # each line contains one sequence
        for line in lines:
            if line.startswith("<source>"):
                new = "<?xml version=\"1.0\"?>\n  <all_news>" + line + "</all_news> "
                # parse xml
                root = ET.fromstring(new)
                cat = root.find("category").text
                if cat in AG_CATEGORIES and count[cat] < AG_CATEGORIES_NUM:
                    title = root.find("title").text
                    desc = root.find("description").text
                    if title and desc:
                        sequ = title + "\n\n" + desc
                        count[cat] += 1
                        sequences.append(sequ)
                        labels.append(create_one_hot_vector(cat,
                                                            AG_CATEGORIES))
                        # print(sequ)
                        # print("-------------")
    return [sequences, labels]


def grab_data_rotten(source_path):
    pos_path = os.path.join(source_path, "rt-polarity.pos")
    neg_path = os.path.join(source_path, "rt-polarity.neg")
    positive_examples = list(open(pos_path).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_path).readlines())
    negative_examples = [s.strip() for s in negative_examples]

    sequences = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    labels = positive_labels + negative_labels

    return [sequences, labels]


def raw_data_statistics(data_name, sequences, labels):
    print("\nStatistics of", data_name, ":")
    print("the nubmer of data examples:", len(sequences))
    print("the average length of data examples:",
          sum(len(s) for s in sequences) / len(sequences))
    print("the number of data examples per category:")
    # list is unhashable type
    # c = collections.Counter(labels)
    labels_str = ['[' + ' '.join(str(b) for b in a) + ']' for a in labels]
    c = collections.Counter(labels_str)
    for key, val in c.items():
        print(key, val)


def load_data_and_labels(dataset, data_dir):
    """return origin sequence vector and label vecto.
    return [sequences, labels]
    sequences: [str]
    labels: [one-hot vectors]
    """
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    if dataset == "rotten":
        source_path = maybe_download(data_dir, ROTTEN_SOURCE,
                                     ROTTEN_DOWNLOADED, ROTTEN_URL)
        # Load data from files
        print("current working directory:", os.getcwd())
        sequences, labels = grab_data_rotten(source_path)
        print("shuffling dataset and splitting train/test sets")
        x_train, y_train, x_test, y_test = shuffle_and_split(sequences, labels,
                                                             test_ratio)

    elif dataset == "ag":
        source_path = maybe_download(data_dir, AG_SOURCE, AG_DOWNLOADED,
                                     AG_URL)

        print("parsing xml file...(it may take a minute)")
        sequences, labels = grab_data_ag(source_path)
        print("sample sequence:", sequences[:10])
        print("sample labels:", labels[:10])
        print("shuffling dataset and splitting train/test sets")
        x_train, y_train, x_test, y_test = shuffle_and_split(sequences, labels,
                                                             test_ratio)

    elif dataset == "newsgroups":
        source_path = maybe_download(data_dir, NEWSGROUPS_SOURCE,
                                     NEWSGROUPS_DOWNLOADED, NEWSGROUPS_URL)

        print("load train set")
        train_path = os.path.join(source_path, "20news-bydate-train")
        x_train, y_train = grab_data_from_folder(
            train_path,
            NEWSGROUPS_CATEGORIES,
            folder_map=lambda x: NEWSGROUPS_DICT.get(x))
        shuffle_data(x_train, y_train)
        print("load test set")
        test_path = os.path.join(source_path, "20news-bydate-test")
        x_test, y_test = grab_data_from_folder(
            test_path,
            NEWSGROUPS_CATEGORIES,
            folder_map=lambda x: NEWSGROUPS_DICT.get(x))
        shuffle_data(x_test, y_test)

    elif dataset == "imdb":
        source_path = maybe_download(data_dir, IMDB_SOURCE, IMDB_DOWNLOADED,
                                     IMDB_URL)
        print("load train set")
        train_path = os.path.join(source_path, "train")
        x_train, y_train = grab_data_from_folder(train_path, IMDB_CATEGORIES)
        shuffle_data(x_train, y_train)
        print("load test set")
        test_path = os.path.join(source_path, "test")
        x_test, y_test = grab_data_from_folder(test_path, IMDB_CATEGORIES)
        shuffle_data(x_test, y_test)

    else:
        print("cannot recognize dataset:", dataset)
        print("example: rotten, ag, newsgroups, imdb.")

    raw_data_statistics("train set", x_train, y_train)
    raw_data_statistics("test set", x_test, y_test)

    return [x_train, y_train, x_test, y_test]


def align_sequences(sequences, padding_character=' ', max_length=-1):
    """Pads or cuts all character-level sequences to the same length.
    The length is defined by the longest sequence if not set.

    sequences -- [str]
    Returns padded sequences.
    """
    print("aligning sequences...")
    if max_length < 0:
        max_length = max(len(x) for x in sequences)
    padded_sequences = []
    for sequence in sequences:
        if not isinstance(sequence, basestring):
            print("error: not string:")
            print(sequence)
            print("\n\n")
        new_sequence = sequence.ljust(max_length,
                                      padding_character)[:max_length]
        padded_sequences.append(new_sequence)
    return padded_sequences

#
# def quantize_sequences(sequences, alphabet):
#     """Giving prescribing alphabet, quantize each caracter using one-hot encoding
#     in each sequence.
#     input:
#         sequences: [str]
#     return:
#         [[[0 or 1]]]
#     """
#     print("quantizing sequences...")
#     feature_size = len(alphabet)
#     new_sequences = []
#     for sequence in sequences:
#         new_sequence = []
#         # add width to fit the conv2D of TF
#         new_sequence.append([])
#         for character in sequence.lower():
#             new_sequence[0].append(create_one_hot_vector(character, alphabet))
#         new_sequences.append(new_sequence)
#
#     return new_sequences

def quantize_sequences(sequences, alphabet):
    """Giving prescribing alphabet, quantize each caracter using index in alphabet
    in each sequence.
    input:
        sequences: [str]
    return:
        [[int]]
    """
    print("quantizing sequences...")
    new_sequences = []
    for sequence in sequences:
        new_sequence = []
        # add width to fit the conv2D of TF
        for character in sequence.lower():
            if character in alphabet:
                new_sequence.append(alphabet.index(character))
            else:
                new_sequence.append(len(alphabet))
        new_sequences.append(new_sequence)

    return new_sequences

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(sequences, labels, name):
    """character level convertion"""
    num_examples = len(labels)
    if len(sequences) != num_examples:
        raise ValueError("sequences size %d does not match label size %d." %
                         (sequences.shape[0], num_examples))

    filename = os.path.join(FLAGS.datasets_dir,
                            FLAGS.dataset +".character"+ '.' + name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(labels[index]),
            'sequence_raw': _bytes_feature(sequences[index].tostring())
        }))
        writer.write(example.SerializeToString())




def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_split(sequences):
    """tokenize and split into words"""
	ret = [clean_split(sequ) for sequ in sequences]
    ret = [sequ.split(" ") for sequ in ret]
    return ret

def align_embedding(sequences, , padding_word="<PAD/>", max_length=-1)
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    print("aligning sequences...")
    if max_length < 0:
        max_length = max(len(x) for x in sequences)
    padded_sequences = []
    for sequence in sequences:
        if not isinstance(sequence, basestring):
            raise ValueError("not string: ", sequence)
        new_sequence = sequence.ljust(max_length,
                                      padding_word)[:max_length]
        padded_sequences.append(new_sequence)
    return padded_sequences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences], dtype=np.uint16 )
    return x

def convert_embed(sequences, labels, name, vocab_size):
    """character level convertion"""
    num_examples = len(labels)
    if len(sequences) != num_examples:
        raise ValueError("sequences size %d does not match label size %d." %
                         (sequences.shape[0], num_examples))

    filename = os.path.join(FLAGS.datasets_dir,
                            FLAGS.dataset +".embedding"+ '.' + name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        example = tf.train.Example(features=tf.train.Features(feature={
            'vocab_size': _int64_feature(vocab_size),
            'label': _int64_feature(labels[index]),
            'sequence_raw': _bytes_feature(sequences[index].tostring())
        }))
        writer.write(example.SerializeToString())

def main(argv):
    # Get the data.
    x_train, y_train, x_test, y_test = load_data_and_labels(FLAGS.dataset,
                                                            FLAGS.datasets_dir)

    if FLAGS.model_type == 'character':
        x_train_alian = align_sequences(x_train, max_length=FLAGS.input_length)
        x_train_quan = quantize_sequences(x_train_alian, alphabet)
        # Convert to Examples and write the result to TFRecords.
        convert_to(np.array(x_train_quan, dtype=np.uint8), [a.index(1) for a in y_train], 'train')

        x_test_alian = align_sequences(x_test, max_length=FLAGS.input_length)
        x_test_quan = quantize_sequences(x_test_alian, alphabet)
        convert_to(np.array(x_test_quan, dtype=np.uint8), [a.index(1) for a in y_test], 'test')

    elif FLAGS.model_type == 'embedding':
        x_train_clean = clean_split(x_train)
        x_train_alian = align_embedding(x_train_clean, max_length=FLAGS.embed_length)
        vocabulary, _ = build_vocab(x_train_alian)
        x_train_build = build_input_data(x_train_alian, vocabulary)
        # Convert to Examples and write the result to TFRecords.
        convert_embed(x_train_build, [a.index(1) for a in y_train], 'train', len(vocabulary))

        x_test_clean = clean_split(x_test)
        x_test_alian = align_embedding(x_test_clean, max_length=FLAGS.embed_length)
        vocabulary, _ = build_vocab(x_test_alian)
        x_test_build = build_input_data(x_test_alian, vocabulary)
        # Convert to Examples and write the result to TFRecords.
        convert_embed(x_test_build, [a.index(1) for a in y_test], 'test', len(vocabulary))

        x_test_alian = align_sequences(x_test, max_length=FLAGS.input_length)
        x_test_quan = quantize_sequences(x_test_alian, alphabet)
        convert_to(np.array(x_test_quan, dtype=np.uint8), [a.index(1) for a in y_test], 'test')
    else:
        print ("error: wrong model_type")


if __name__ == '__main__':
    tf.app.run()

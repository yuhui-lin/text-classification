"""This code process TFRecords text classification datasets.
YOU MUST run convert_data before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dataset", "rotten",
    "dataset used to train the neural network: rotten, ag, newsgroups, imdb. (default: rotten_tomato)")
tf.app.flags.DEFINE_string('datasets_dir', 'datasets',
                           'Directory to download data files and write the '
                           'converted result')
#tf.app.flags.DEFINE_string('datasets_dir', '~/Downloads/text-classification',
                           #'Directory to download data files and write the '
                           #'converted result')
tf.app.flags.DEFINE_integer(
    "input_length", 1014,
    "number of characters in each input sequences (default: 1024)")
tf.app.flags.DEFINE_integer("alphabet_length", 71,
                            "number of characters in aphabet (default: 71)")

# Constants used for dealing with the files, matches convert_to_records.
tfrecord_suffix = '.tfrecords'


def char_index_batch_to_2d_tensor(batch, batch_size, num_labels):
    sparse_labels = tf.reshape(batch, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    concatenated = tf.concat(1, [indices, sparse_labels])
    concat = tf.concat(0, [[batch_size], [num_labels]])
    output_shape = tf.reshape(concat, [2])
    sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1, 0)
    return tf.reshape(sparse_to_dense, [batch_size, num_labels])


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.FixedLenFeature(
                [], tf.int64),
            'sequence_raw': tf.FixedLenFeature(
                [], tf.string),
        })
    sequence = features['sequence_raw']

    # preprocess
    s_decode = tf.decode_raw(sequence, tf.uint8)
    s_batch = tf.cast(s_decode, tf.int32)
    s_encode = char_index_batch_to_2d_tensor(s_batch, FLAGS.input_length,
                                             FLAGS.alphabet_length + 1)
    s_expand = tf.expand_dims(s_encode, 0)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    s_cast = tf.cast(s_expand, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return s_cast, label


def inputs_character(datatype, batch_size, num_epochs=None, min_shuffle=1):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    filename = os.path.abspath(os.path.join(
        FLAGS.datasets_dir,
        FLAGS.dataset + ".character" + '.' + datatype + tfrecord_suffix))
    print("Reading examples from file: {}\n".format(filename))

    with tf.name_scope('inputs_character'):
        filename_queue = tf.train.string_input_producer(
            [filename],
            num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        sequence, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        capacity = min_shuffle + 3 * batch_size
        sequences, sparse_labels = tf.train.shuffle_batch(
            [sequence, label],
            batch_size=batch_size,
            num_threads=2,
            capacity=capacity,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=min_shuffle)

        return sequences, sparse_labels

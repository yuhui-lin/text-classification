"""This code process TFRecords text classification datasets.
YOU MUST run convert_data before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
import numpy as np

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dataset", "rotten",
    "dataset used to train the neural network: rotten, ag, newsgroups, imdb. (default: rotten_tomato)")
tf.app.flags.DEFINE_string('datasets_dir', 'datasets',
                           'Directory to download data files and write the '
                           'converted result')

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=0
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=0
##tf.app.flags.DEFINE_integer(
    #"input_length", 1014,
    #"number of characters in each input sequences (default: 1024)")
#tf.app.flags.DEFINE_integer("alphabet_length", 71,
                            #"number of characters in aphabet (default: 71)")

# Constants used for dealing with the files, matches convert_to_records.
tfrecord_suffix = '.tfrecords'


#def char_index_batch_to_2d_tensor(batch, batch_size, num_labels):
    #sparse_labels = tf.reshape(batch, [batch_size, 1])
    #indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    #concatenated = tf.concat(1, [indices, sparse_labels])
    #concat = tf.concat(0, [[batch_size], [num_labels]])
    #output_shape = tf.reshape(concat, [2])
    #sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1, 0)
    #return tf.reshape(sparse_to_dense, [batch_size, num_labels])


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
    # preprocess
    #s_decode = tf.decode_raw(features['sequence_raw'], tf.uint8)
    s_decode = tf.decode_raw(features['sequence_raw'], tf.int32)
    s_decode.set_shape([DOC_LEN])

    #s_batch = tf.cast(s_decode, tf.int32)
    #s_encode = char_index_batch_to_2d_tensor(s_batch, FLAGS.input_length,
                                             #FLAGS.alphabet_length + 1)
    #s_expand = tf.expand_dims(s_encode, 0)
    #s_expand = tf.expand_dims(s_encode, 2)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    #s_cast = tf.cast(s_expand, tf.float32)
    s_cast = tf.cast(s_decode, tf.int32)
    label = tf.cast(features['label'], tf.int32)

    return s_cast, label

def get_doclen_voclen():
    global DOC_LEN, VOC_LEN
    with open(os.path.join(FLAGS.datasets_dir, 'meta'), 'r') as f:
        DOC_LEN = int(f.readline().strip())
        VOC_LEN = int(f.readline().strip())
    return DOC_LEN, VOC_LEN

DOC_LEN, VOC_LEN = get_doclen_voclen()

def inputs(datatype, batch_size, num_epochs=None, min_shuffle=1):
    """Reads input data num_epochs times.
    Args:
      datatype: Selects between the training (True) and test (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
      A tuple (sequence, labels), where:
      * seqeuences is a float tensor with shape [batch_size, len of doc]
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """

    filename = os.path.abspath(os.path.join(
        FLAGS.datasets_dir,
        FLAGS.dataset + ".word" + '.' + datatype + tfrecord_suffix))
    print("Reading examples from file: {}\n".format(filename))
    if datatype=='train':
      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN  
    elif datatype == 'test':
      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL 
    else:
      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    with tf.name_scope('inputs_word'):
        filename_queue = tf.train.string_input_producer(
            [filename],
            num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        sequence, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int( num_examples_per_epoch *
                                         min_fraction_of_examples_in_queue)
        min_shuffle = min_queue_examples
        capacity = min_shuffle + 3 * batch_size
        sequences, sparse_labels = tf.train.shuffle_batch(
            [sequence, label],
            batch_size=batch_size,
            num_threads=2,
            capacity=capacity,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=min_shuffle)

        return sequences, sparse_labels



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

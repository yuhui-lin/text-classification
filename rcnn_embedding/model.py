"""CNN model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
from tensorflow.python.ops import array_ops as array_ops_

import inputs

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer("num_epochs", 100,
                            "Number of training epochs (default: 100)")
tf.app.flags.DEFINE_integer("minibatch_size", 128,
                            "mini Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("word_d", 50,
                            "word vector dimension")
tf.app.flags.DEFINE_integer("num_layers", 1,
                            "word vector dimension")
tf.app.flags.DEFINE_integer("hidden_layers", 50,
                            "word vector dimension")
tf.app.flags.DEFINE_integer("num_local", 1024,
                            "word vector dimension")

# global constants
# ==================================
# this three dataset related variable are initialized in train.main
# NUM_CLASSES = 0
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
# output num for conv layer
FEATURE_NUM = 256

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def initial_dataset_info(dataset):
    if dataset == "rotten":
        NUM_CLASSES = 2
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8530
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2132
    elif dataset == "ag":
        NUM_CLASSES = 4
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
    elif dataset == "newsgroups":
        NUM_CLASSES = 4
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
    elif dataset == "imdb":
        NUM_CLASSES = 2
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
        NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
    else:
        print("wrong dataset:",dataset)
        return False
    return True
def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = _variable_on_cpu(name,
                           shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inputs_train():
    """Construct input examples for training process.
    Returns:
        sequences: 4D tensor of [batch_size, 1, input_length, alphabet_length] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    return inputs.inputs_embedding("train",
                                   FLAGS.minibatch_size,
                                   FLAGS.num_epochs,
                                   min_shuffle=1000)


def inputs_eval():
    """Construct input examples for evaluations.
    similar to inputs_train
    """
    # don't shuffle
    return inputs.inputs_embedding("test",
                                   FLAGS.minibatch_size,
                                   None,
                                   min_shuffle=1)


def _reverse_seq(input_seq, lengths):
    """Reverse a list of Tensors up to specified lengths.
    Args:
        input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
        lengths:   A tensor of dimension batch_size, containing lengths for each
                   sequence in the batch. If "None" is specified, simply reverses
                   the list.
    Returns:
        time-reversed sequence
    """
    for input_ in input_seq:
        input_.set_shape(input_.get_shape().with_rank(2))

    # Join into (time, batch_size, depth)
    s_joined = array_ops_.pack(input_seq)

    # Reverse along dimension 0
    s_reversed = array_ops_.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = array_ops_.unpack(s_reversed)
    return result


def get_embedding(sequences):
    embedding = inputs.get_embedding()
    with tf.device('/cpu:0'), tf.variable_scope("embedding"):
        W = tf.constant(embedding, name="W")
        # WW = tf.reshape(W, [-1, 50])
        # W = tf.squeeze(WW)
        # W = tf.cast(WW, tf.float32)
        # print("shape;", sequences.dtype)
        embedded_chars = tf.nn.embedding_lookup(W, sequences)
    # embedded_squeeze = tf.squeeze(embedded_chars, [0])
    # [100, 128, 50]
    inputs_rnn = [tf.squeeze(input_, [1])
            for input_ in tf.split(1, FLAGS.embed_length, embedded_chars)]
    # inputs_rnn = tf.transpose(embedded_chars, perm=[1,0,2])
    with tf.variable_scope("BiRNN_FW"):
        cell_fw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_layers)
        cells_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * FLAGS.num_layers)
        initial_state_fw = cells_fw.zero_state(FLAGS.minibatch_size, tf.float32)
        outputs_fw, state_fw = tf.nn.rnn(cells_fw, inputs_rnn, initial_state=initial_state_fw)
        # _activation_summary(outputs_fw)

    with tf.variable_scope("BiRNN_BW") as scope:
        cell_bw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_layers)
        cells_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * FLAGS.num_layers)
        initial_state_bw = cells_bw.zero_state(FLAGS.minibatch_size, tf.float32)
        outputs_tmp, state_bw = tf.nn.rnn(cells_bw, inputs_rnn[::-1], initial_state=initial_state_bw)
        # [100, 128, 50]
        # output_bw = _reverse_seq(outputs_tmp, FLAGS.embed_length)
        outputs_bw = outputs_tmp[::-1]
        # _activation_summary(outputs_bw)

    with tf.variable_scope('concat') as scope:

        # [100, 128, 150]
        # change list to tensor
        outputs = tf.concat(2, [tf.pack(tensor) for tensor in [outputs_fw, inputs_rnn, outputs_bw]])
        # -> [128, 100, 150] -> [-1, 150]
        dim = FLAGS.word_d+2*FLAGS.hidden_layers
        xi = tf.reshape(tf.transpose(outputs, perm=[1,0,2]), [-1, dim])

    # local1
    with tf.variable_scope('local1') as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, 1024],
                                              stddev=0.02,
                                              wd=None)
        biases = _variable_on_cpu('biases', [1024],
                                  tf.constant_initializer(0.02))
        local1 = tf.nn.tanh(
            tf.matmul(xi, weights) + biases,
            name=scope.name)
        _activation_summary(local1)

    local1_reshape = tf.reshape(local1, [FLAGS.minibatch_size, FLAGS.embed_length, 1024])
    local1_expand = tf.expand_dims(local1_reshape, -1)
    # pool1
    pool1 = tf.nn.max_pool(local1_expand,
                           ksize=[1, FLAGS.embed_length, 1, 1],
                           strides=[1, 1, 1, 1],
                           padding='VALID',
                           name='pool1')

    # [128, 1024]
    pool1_squeeze = tf.squeeze(pool1)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        print ("NUM_CLASSES:", NUM_CLASSES)
        weights = _variable_with_weight_decay('weights',
                                              [FLAGS.num_local, NUM_CLASSES],
                                              stddev=0.02,
                                              wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.02))
        softmax_linear = tf.add(
            tf.matmul(pool1_squeeze, weights),
            biases,
            name=scope.name)
        _activation_summary(softmax_linear)
    return softmax_linear


def inference(sequences):
    """Build the RCNN model.
    Args:
        sequences: Sequences returned from inputs_train() or inputs_eval.
    Returns:
        Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables for both training and
    # evaluation.
    embedding = inputs.get_embedding()
    with tf.device('/cpu:0'), tf.variable_scope("embedding"):
        W = tf.constant(embedding, name="W")
        # WW = tf.reshape(W, [-1, 50])
        # W = tf.squeeze(WW)
        # W = tf.cast(WW, tf.float32)
        # print("shape;", sequences.dtype)
        embedded_chars = tf.nn.embedding_lookup(W, sequences)
    # embedded_squeeze = tf.squeeze(embedded_chars, [0])
    # [100, 128, 50]
    inputs_rnn = [tf.squeeze(input_, [1])
            for input_ in tf.split(1, FLAGS.embed_length, embedded_chars)]
    # inputs_rnn = tf.transpose(embedded_chars, perm=[1,0,2])
    with tf.variable_scope("BiRNN_FW"):
        cell_fw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_layers)
        cells_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * FLAGS.num_layers)
        initial_state_fw = cells_fw.zero_state(FLAGS.minibatch_size, tf.float32)
        outputs_fw, state_fw = tf.nn.rnn(cells_fw, inputs_rnn, initial_state=initial_state_fw)
        # _activation_summary(outputs_fw)

    with tf.variable_scope("BiRNN_BW") as scope:
        cell_bw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_layers)
        cells_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * FLAGS.num_layers)
        initial_state_bw = cells_bw.zero_state(FLAGS.minibatch_size, tf.float32)
        outputs_tmp, state_bw = tf.nn.rnn(cells_bw, inputs_rnn[::-1], initial_state=initial_state_bw)
        # [100, 128, 50]
        # output_bw = _reverse_seq(outputs_tmp, FLAGS.embed_length)
        outputs_bw = outputs_tmp[::-1]
        # _activation_summary(outputs_bw)

    with tf.variable_scope('concat') as scope:

        # [100, 128, 150]
        # change list to tensor
        outputs = tf.concat(2, [tf.pack(tensor) for tensor in [outputs_fw, inputs_rnn, outputs_bw]])
        # -> [128, 100, 150] -> [-1, 150]
        dim = FLAGS.word_d+2*FLAGS.hidden_layers
        xi = tf.reshape(tf.transpose(outputs, perm=[1,0,2]), [-1, dim])

    # local1
    with tf.variable_scope('local1') as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, 1024],
                                              stddev=0.02,
                                              wd=None)
        biases = _variable_on_cpu('biases', [1024],
                                  tf.constant_initializer(0.02))
        local1 = tf.nn.tanh(
            tf.matmul(xi, weights) + biases,
            name=scope.name)
        _activation_summary(local1)

    local1_reshape = tf.reshape(local1, [FLAGS.minibatch_size, FLAGS.embed_length, 1024])
    local1_expand = tf.expand_dims(local1_reshape, -1)
    # pool1
    pool1 = tf.nn.max_pool(local1_expand,
                           ksize=[1, FLAGS.embed_length, 1, 1],
                           strides=[1, 1, 1, 1],
                           padding='VALID',
                           name='pool1')

    # [128, 1024]

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        pool1_squeeze = tf.squeeze(pool1)
        print ("NUM_CLASSES:", NUM_CLASSES)
        weights = _variable_with_weight_decay('weights',
                                              [FLAGS.num_local, NUM_CLASSES],
                                              stddev=0.02,
                                              wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.02))
        softmax_linear = tf.add(
            tf.matmul(pool1_squeeze, weights),
            biases,
            name=scope.name)
        _activation_summary(softmax_linear)
    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits,
        labels,
        name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CNN model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def training(total_loss, global_step):
    """Train CNN model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.minibatch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr_decay = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                          global_step,
                                          decay_steps,
                                          LEARNING_RATE_DECAY_FACTOR,
                                          staircase=True)
    # compare with 0.01 * 0.5^10
    lr = tf.maximum(lr_decay, 0.000009765625)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
    #                                                       global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op

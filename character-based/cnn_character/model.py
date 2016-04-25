"""CNN model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

import inputs

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer("num_epochs", 100,
                            "Number of training epochs (default: 100)")
tf.app.flags.DEFINE_integer("minibatch_size", 128,
                            "mini Batch Size (default: 64)")

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
        print("wrong dataset")
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
    return inputs.inputs_character("train",
                                   FLAGS.minibatch_size,
                                   FLAGS.num_epochs,
                                   min_shuffle=1000)


def inputs_eval():
    """Construct input examples for evaluations.
    similar to inputs_train
    """
    # don't shuffle
    return inputs.inputs_character("test",
                                   FLAGS.minibatch_size,
                                   None,
                                   min_shuffle=1)


def inference(sequences):
    """Build the CNN model.
    Args:
        sequences: Sequences returned from inputs_train() or inputs_eval.
    Returns:
        Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables for both training and
    # evaluation.

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[1, 7, FLAGS.alphabet_length + 1, FEATURE_NUM],
            stddev=0.02,
            wd=None)
        conv = tf.nn.conv2d(sequences, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FEATURE_NUM],
                                  tf.constant_initializer(0.02))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 1, 3, 1],
                           strides=[1, 1, 3, 1],
                           padding='SAME',
                           name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[1, 7, FEATURE_NUM, FEATURE_NUM],
            stddev=0.02,
            wd=None)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FEATURE_NUM],
                                  tf.constant_initializer(0.02))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 1, 3, 1],
                           strides=[1, 1, 3, 1],
                           padding='SAME',
                           name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[1, 3, FEATURE_NUM, FEATURE_NUM],
            stddev=0.02,
            wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FEATURE_NUM],
                                  tf.constant_initializer(0.02))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[1, 3, FEATURE_NUM, FEATURE_NUM],
            stddev=0.02,
            wd=None)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FEATURE_NUM],
                                  tf.constant_initializer(0.02))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[1, 3, FEATURE_NUM, FEATURE_NUM],
            stddev=0.02,
            wd=None)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FEATURE_NUM],
                                  tf.constant_initializer(0.02))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv5)

    # conv6
    with tf.variable_scope('conv6') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[1, 3, FEATURE_NUM, FEATURE_NUM],
            stddev=0.02,
            wd=None)
        conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [FEATURE_NUM],
                                  tf.constant_initializer(0.02))
        bias = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv6)

    # pool6
    pool6 = tf.nn.max_pool(conv6,
                           ksize=[1, 1, 3, 1],
                           strides=[1, 1, 3, 1],
                           padding='SAME',
                           name='pool6')

    # local7
    with tf.variable_scope('local7') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool6, [FLAGS.minibatch_size, -1])
        dim = reshape.get_shape()[1].value

        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, 1024],
                                              stddev=0.02,
                                              wd=None)
        biases = _variable_on_cpu('biases', [1024],
                                  tf.constant_initializer(0.02))
        local7 = tf.nn.relu(
            tf.matmul(reshape, weights) + biases,
            name=scope.name)
        _activation_summary(local7)

    # dropout7
    dropout7 = tf.nn.dropout(local7, FLAGS.dropout_keep_prob, name="dropout7")

    # local8
    with tf.variable_scope('local8') as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[1024, 1024],
                                              stddev=0.02,
                                              wd=None)
        biases = _variable_on_cpu('biases', [1024],
                                  tf.constant_initializer(0.02))
        local8 = tf.nn.relu(
            tf.matmul(dropout7, weights) + biases,
            name=scope.name)
        _activation_summary(local8)

    # dropout8
    dropout8 = tf.nn.dropout(local8, FLAGS.dropout_keep_prob, name="dropout8")

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        print ("NUM_CLASSES:", NUM_CLASSES)
        weights = _variable_with_weight_decay('weights',
                                              [1024, NUM_CLASSES],
                                              stddev=0.02,
                                              wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.02))
        softmax_linear = tf.add(
            tf.matmul(dropout8, weights),
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

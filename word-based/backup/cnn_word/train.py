from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf

# from cnn_character.model import model
from cnn_word import model
import proc_data
import inputs
#import model

# Parameters
# ==================================================
FLAGS = tf.app.flags.FLAGS

# Model Hyperparameters
# tf.app.flags.DEFINE_integer(
#     "input_length", 1014,
#     "number of characters in each input sequences (default: 1014)")
# Training parameters
tf.app.flags.DEFINE_integer(
    "evaluate_every", 100,
    "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100,
                            "Save model after this many steps (default: 100)")
# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                            "Allow device soft device placement")
# tf.app.flags.DEFINE_boolean("log_device_placement", False,
#                             "Log placement of ops on devices")

tf.app.flags.DEFINE_string('outputs_dir', 'cnn_word/logs', 'output dir for summary and checkpoints')

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('summary_step', 25,
                            """Number of steps to write summaries.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 50,
                            """Number of steps to write checkpoint. """)
tf.app.flags.DEFINE_integer('print_step', 10,
                            """Number of steps to print. """)
#tf.app.flags.DEFINE_string('dataset', 'rotten',
                           #"""data set""")
# ==================================================
# Output directory for checkpoints and summaries
#timestamp = FLAGS.dataset + '.' + str(datetime.now().strftime("%Y-%m-%d.%H-%M-%S"))
#TRAIN_DIR = os.path.abspath(c)
#SUMMARY_DIR = os.path.join(TRAIN_DIR, "summaries")
#CHECKPOINT_DIR = os.path.join(TRAIN_DIR, "checkpoints")
#CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model.ckpt')
timestamp = ''
TRAIN_DIR = ''
SUMMARY_DIR = ''
CHECKPOINT_DIR = ''
CHECKPOINT_PATH = ''
# functions
# ==================================================

def train2():
    """Train CNN for a number of steps."""
    print("start training...")
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        x_train, y_train, x_test, y_test, vocab = proc_data.load_data_and_labels()

        y_train = np.array([1 if y[1]==1 else 0 for y in y_train])
        y_test = np.array([1 if y[1]==1 else 0  for y in y_test])
        print(x_train.shape, 'x_train shape')
        print(y_train.shape, 'y_train shape')
        print(x_test.shape, 'x_test shape')
        print(y_test.shape, 'y_test shape')
        doc_lenth = len(x_train[0]) 
        # get input data
        #sequences, labels = model.inputs_train()
        sequences = tf.placeholder(tf.int32, [FLAGS.minibatch_size, doc_lenth], name = 'input_x')
        labels = tf.placeholder(tf.int32, [FLAGS.minibatch_size], name = 'input_y')
        dropout_keep_prob = tf.placeholder(tf.float32, name='drop_out')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        print(tf.shape(sequences),'sequences')
        logits = model.inference(sequences, doc_lenth, len(vocab), dropout_keep_prob)

        # Calculate loss.
        loss = model.loss(logits, labels)

        # accuracy
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.reduce_sum(tf.cast(top_k_op, tf.int32))

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.training(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=
                                                FLAGS.log_device_placement))
        # with tf.Session(config=tf.ConfigProto(
        #     log_device_placement=FLAGS.log_device_placement)) as sess:
        sess.run(init)

        # Start the queue runners.
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)


        # Generate batches
        batches = inputs.batch_iter(
            list(zip(x_train, y_train)), FLAGS.minibatch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        step = 0
        for batch in batches:
            step+=1
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            print(x_batch.shape, 'x_batch shape', step)
            print(y_batch.shape, 'y_batch shape', step)
            feed_dict = {
              sequences: x_batch,
              labels: y_batch,
              dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss], feed_dict)
            end_time = time.time()
            duration = end_time - start_time
 
              # print current state
            if step % FLAGS.print_step == 0:
                num_examples_per_step = FLAGS.minibatch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # save summary
            if step % FLAGS.summary_step == 0:
                summary_str = sess.run(summary_op, feed_dict)
                summary_writer.add_summary(summary_str, step)
                print("step: {}, wrote summaries.".format(step))

            # Save the model checkpoint periodically.
            if step % FLAGS.checkpoint_step == 0 or (
                    step + 1) == FLAGS.max_steps:
                saver_path = saver.save(sess,
                                        CHECKPOINT_PATH,
                                        global_step=step)
                print("\nSaved model checkpoint to {}\n".format(
                    saver_path))
        
        # start evaluation for this checkpoint
        all_x = np.concatenate((x_test, x_train), axis=0)
        all_y = np.concatenate((y_test, y_train), axis=0)
        #batches = inputs.batch_iter(
            #list(zip(x_test, y_test)), FLAGS.minibatch_size, 1)
        
        batches = inputs.batch_iter(
            list(zip(all_x, all_y)), FLAGS.minibatch_size, 1)

        num_batches = len(all_y) // FLAGS.minibatch_size
        true_count = 0
        for batch in batches:
            step+=1
            x_batch, y_batch = zip(*batch)
            #train_step(x_batch, y_batch)
            feed_dict = {
              sequences: x_batch,
              labels: y_batch,
              dropout_keep_prob: 1.0
            }
            _, cor = sess.run([ top_k_op, correct ], feed_dict)
            print("get %d correct out of %d" %(cor, FLAGS.minibatch_size))
            true_count += cor
 
        accuracy = 100.0 *true_count / (num_batches * FLAGS.minibatch_size)
        print("Accuracy of test: %.2f%%\n"%accuracy)

def train():
    """Train CNN for a number of steps."""
    print("start training...")
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # get input data
        #sequences, labels = model.inputs_train()
        sequences = tf.placeholder(tf.int32, [None, batch_size], name = 'input_x')
        labels = tf.placeholder(tf.int32, [None, model.NUM_CLASSES], name = 'input_x')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(sequences)

        # Calculate loss.
        loss = model.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.training(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=
                                                FLAGS.log_device_placement))
        # with tf.Session(config=tf.ConfigProto(
        #     log_device_placement=FLAGS.log_device_placement)) as sess:
        sess.run(init)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

        try:
            step = 1
            while not coord.should_stop():
                start_time = time.time()
                #_, loss_value = sess.run([train_op, loss])
                seq, lab = sess.run([sequences, labels])
                duration = time.time() - start_time
                print(seq)
                print(lab)

                assert not np.isnan(
                    loss_value), 'Model diverged with loss = NaN'

                ## print current state
                #if step % FLAGS.print_step == 0:
                    #num_examples_per_step = FLAGS.minibatch_size
                    #examples_per_sec = num_examples_per_step / duration
                    #sec_per_batch = float(duration)

                    #format_str = (
                        #'%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        #'sec/batch)')
                    #print(format_str % (datetime.now(), step, loss_value,
                                        #examples_per_sec, sec_per_batch))

                ## save summary
                #if step % FLAGS.summary_step == 0:
                    #summary_str = sess.run(summary_op)
                    #summary_writer.add_summary(summary_str, step)
                    #print("step: {}, wrote summaries.".format(step))

                ## Save the model checkpoint periodically.
                #if step % FLAGS.checkpoint_step == 0 or (
                        #step + 1) == FLAGS.max_steps:
                    #saver_path = saver.save(sess,
                                            #CHECKPOINT_PATH,
                                            #global_step=step)
                    #print("\nSaved model checkpoint to {}\n".format(
                        #saver_path))
                    ## start evaluation for this checkpoint

                step += 1
                time.sleep(1)

        except tf.errors.OutOfRangeError:
            print("Done~")
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    print("start of main")

    timestamp = FLAGS.dataset + '.' + str(datetime.now().strftime("%Y-%m-%d.%H-%M-%S"))
    c = os.path.join(FLAGS.outputs_dir, timestamp)
    TRAIN_DIR = os.path.abspath(c)
    SUMMARY_DIR = os.path.join(TRAIN_DIR, "summaries")
    CHECKPOINT_DIR = os.path.join(TRAIN_DIR, "checkpoints")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model.ckpt')

    print("\nParameters:")
    # weird bug: add this to enable print all Parameters.
    # FLAGS.minibatch_size
    for attr, value in sorted(FLAGS.__flags.iteritems()):
        print("{}={}".format(attr.upper(), value))
    print("")

    if FLAGS.dataset == "rotten":
        model.NUM_CLASSES = 2
        model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8530
        model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2132
    elif FLAGS.dataset == "ag":
        model.NUM_CLASSES = 4
        model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
        model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
    elif FLAGS.dataset == "newsgroups":
        model.NUM_CLASSES = 4
        model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
        model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
    elif FLAGS.dataset == "imdb":
        model.NUM_CLASSES = 2
        model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
        model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
    else:
        print("wrong dataset")
    # model.initial_dataset_info(FLAGS.dataset)

    if not tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.MakeDirs(TRAIN_DIR)
    print("\nWriting to {}\n".format(TRAIN_DIR))

    if not tf.gfile.Exists(CHECKPOINT_DIR):
        tf.gfile.MakeDirs(CHECKPOINT_DIR)

    train2()

    print("\n end of main")


if __name__ == '__main__':
    tf.app.run()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf

# from cnn_character.model import model
from rcnn_embedding import model

# Parameters
# ==================================================
FLAGS = tf.app.flags.FLAGS

# Model Hyperparameters
# tf.app.flags.DEFINE_integer(
#     "input_length", 1014,
#     "number of characters in each input sequences (default: 1014)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "Dropout keep probability (default: 0.5)")

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

tf.app.flags.DEFINE_string('outputs_dir', 'cnn_character/outputs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('print_step', 1,
                            """Number of steps to print current state.""")
tf.app.flags.DEFINE_integer('summary_step', 3,
                            """Number of steps to write summaries.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 50,
                            """Number of steps to write checkpoint. """)
tf.app.flags.DEFINE_integer('num_checkpoints', 10,
                            """Number of maximum checkpoints to keep. default: 10""")

# ==================================================
# Output directory for checkpoints and summaries
timestamp = FLAGS.dataset + '.' + str(datetime.now().strftime("%Y-%m-%d.%H-%M-%S"))
TRAIN_DIR = os.path.abspath(os.path.join(FLAGS.outputs_dir, timestamp))
SUMMARY_DIR = os.path.join(TRAIN_DIR, "summaries")
CHECKPOINT_DIR = os.path.join(TRAIN_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model.ckpt')

# functions
# ==================================================


def train():
    """Train CNN for a number of steps."""
    print("start training...")
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # get input data
        sequences, labels = model.inputs_train()
        # logits = model.get_embedding(sequences)

        # Build a Graph that computes the logits predictions from the
        logits = model.inference(sequences)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Calculate loss.
        loss = model.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.training(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.num_checkpoints)

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
                _, loss_value, top_k = sess.run([train_op, loss, top_k_op])
                # l = sess.run([logits])
                # l = sess.run([emb])
                duration = time.time() - start_time
                # print("shape embedding:", np.array(l).shape)
                # print(l[0][0][-1])

                assert not np.isnan(
                    loss_value), 'Model diverged with loss = NaN'
                #
                # print current state
                if step % FLAGS.print_step == 0:
                    num_examples_per_step = FLAGS.minibatch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    precision = np.sum(top_k) / FLAGS.minibatch_size
                    format_str = (
                        '%s: step %d, loss = %.2f, precision = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, precision,
                                        examples_per_sec, sec_per_batch))

                # save summary
                if step % FLAGS.summary_step == 0:
                    summary_str = sess.run(summary_op)
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

                step += 1
                # sleep for test use
                # print("sleep 1 second...")
                # time.sleep(1)

        except tf.errors.OutOfRangeError:
            print("Done~")
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    print("start of main")
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
        model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 128000
        model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 32000
    elif FLAGS.dataset == "newsgroups":
        model.NUM_CLASSES = 4
        model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8356
        model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5563
    elif FLAGS.dataset == "imdb":
        model.NUM_CLASSES = 2
        model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 25000
        model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 25000
    else:
        print("wrong dataset")
    # model.initial_dataset_info(FLAGS.dataset)

    if not tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.MakeDirs(TRAIN_DIR)
    print("\nWriting to {}\n".format(TRAIN_DIR))

    if not tf.gfile.Exists(CHECKPOINT_DIR):
        tf.gfile.MakeDirs(CHECKPOINT_DIR)

    train()

    print ("summary dir:", SUMMARY_DIR)
    print("\n end of main")


if __name__ == '__main__':
    tf.app.run()

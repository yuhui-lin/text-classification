"""Evaluation for RCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf

from rcnn_embedding import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'outputs/',
                           """Directory where to read raining results.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """number of examples per batch.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

tf.app.flags.DEFINE_float("dropout_keep_prob", 1,
                          "Dropout keep probability (default: 1)")

# glogbal parameters
# ===============================
CHECKPOINT_DIR = os.path.join(FLAGS.train_dir, "checkpoints")
EVAL_DIR = os.path.join(FLAGS.train_dir, "eval-" + str(int(time.time())))


# functions
# ===============================
def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[
                -1]
            print("\nglobal step:", global_step)
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess,
                                                 coord=coord,
                                                 daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL /
                                     FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
            print("write eval summary")
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CNN for a number of steps."""
    with tf.Graph().as_default() as g, tf.device("/cpu:0"):
        # Get sequences and labels
        sequences, labels = model.inputs_eval()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(sequences)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     model.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(EVAL_DIR, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                print("eval only once, stope eval")
                break
            print("sleep for {} seconds".format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(CHECKPOINT_DIR):
        print ("train_dir:", os.path.abspath(FLAGS.train_dir))
        train_dir_name = os.path.basename(os.path.abspath(FLAGS.train_dir))
        print ("train_name:", train_dir_name)
        dataset = train_dir_name.split('.')[0]
        print("dataset:", dataset)
        # if not model.initial_dataset_info(dataset):
        #     return

        if dataset == "rotten":
            model.NUM_CLASSES = 2
            model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8530
            model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2132
        elif dataset == "ag":
            model.NUM_CLASSES = 4
            model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
            model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
        elif dataset == "newsgroups":
            model.NUM_CLASSES = 4
            model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
            model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
        elif dataset == "imdb":
            model.NUM_CLASSES = 2
            model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0
            model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
        else:
            print("wrong dataset:", dataset)
            return

        if tf.gfile.Exists(EVAL_DIR):
            tf.gfile.DeleteRecursively(EVAL_DIR)
        tf.gfile.MakeDirs(EVAL_DIR)
        evaluate()
    else:
        print("error: cannot find checkpoints directory!")


if __name__ == '__main__':
    tf.app.run()

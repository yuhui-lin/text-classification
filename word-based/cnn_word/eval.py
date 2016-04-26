"""Evaluation for CNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf

from cnn_word import model 
import inputs
#import proc_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'cnn_word/logs/',
                           """Directory where to write training results.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '',
                           """Directory where to read training results in order to evaluate.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """number of examples per batch.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

tf.app.flags.DEFINE_integer('num_examples', 1000,
                            """Number of examples to run.""")

#CWD = os.path.dirname(os.path.abspath(__file__))
# glogbal parameters
# ===============================
#CHECKPOINT_DIR = os.path.join(FLAGS.train_dir, "checkpoints")
#EVAL_DIR = os.path.join(FLAGS.train_dir, "eval-" + str(int(time.time())))
CHECKPOINT_DIR = ''
EVAL_DIR =''

# functions
# ===============================
def eval_once(saver, summary_writer, top_k_op, summary_op, dropout_keep_prob):
    """Run Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.join(CHECKPOINT_DIR, 'checkpoints'))
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

            num_iter = int(math.ceil(FLAGS.num_examples /
                                     FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run(top_k_op, feed_dict = {dropout_keep_prob: 1.0})
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op, feed_dict = {dropout_keep_prob: 1.0}))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
            print("write eval summary")
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate2():

  # Load data. Load your own data here
  print("Loading data...")
  x1, y1, x2, y2, vocab = proc_data.load_data_and_labels()
  y1 = np.array([1 if y[1]==1 else 0 for y in y1])
  y2 = np.array([1 if y[1]==1 else 0  for y in y2])
  all_x = np.concatenate((x1, x2), axis=0)
  all_y = np.concatenate((y1, y2), axis=0)
  doc_lenth = len(all_x[0]) 

  # Evaluation
  # ==================================================
  checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.checkpoint_dir, 'checkpoints'))
  print('checkpoint_file', checkpoint_file)
  graph = tf.Graph()
  with graph.as_default():
      #session_conf = tf.ConfigProto(
        #allow_soft_placement=FLAGS.allow_soft_placement,
        #log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session()
      with sess.as_default():
          saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
          saver.restore(sess, checkpoint_file)
          
          # Get the placeholders from the graph by name
          sequences = graph.get_operation_by_name("input_x").outputs[0]
          
          # input_y = graph.get_operation_by_name("input_y").outputs[0]
          dropout_keep_prob = graph.get_operation_by_name("drop_out").outputs[0]
          
          # Tensors we want to evaluate
          logits = graph.get_operation_by_name("softmax/logits").outputs[0]

          # Get the placeholders from the graph by name
          #sequences = graph.get_operation_by_name("input_x").outputs[0]
          #labels = graph.get_operation_by_name("input_y").outputs[0]
          #dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

          # Tensors we want to evaluate
          #predictions = graph.get_operation_by_name("output/predictions").outputs[0]

          # Generate batches for one epoch
          print("\nEvaluating...\n")
          batches = inputs.batch_iter(
              list(zip(all_x, all_y)), FLAGS.minibatch_size, 1)

          num_batches = len(all_y) // FLAGS.minibatch_size
          true_count = 0
          step=0
          for batch in batches:
              step+=1
              x_batch, y_batch = zip(*batch)
              x_batch = np.array(x_batch)
              y_batch = np.array(y_batch)
              #train_step(x_batch, y_batch)
              feed_dict = {
                sequences: x_batch,
                dropout_keep_prob: 1.0
              }
              scores = sess.run( logits , feed_dict = feed_dict)
              prediction = np.argmax(scores, axis=1)
              correct = float(sum(prediction == y_batch))

              print("get %d correct out of %d" %(correct, FLAGS.minibatch_size))
              true_count += correct
         
          accuracy = 100.0 *true_count / (num_batches * FLAGS.minibatch_size)
          print("Accuracy of test: %.2f%%\n"%accuracy)



def evaluate():
    """Eval CNN for a number of steps."""
    with tf.Graph().as_default() as g, tf.device("/cpu:0"):
        # Get sequences and labels
        sequences, labels = model.inputs_eval(way=0)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        #logits = model.inference(sequences)
        dropout_keep_prob = tf.placeholder(tf.float32, name='drop_out')
        logits = model.inference(sequences, inputs.DOC_LEN, inputs.VOC_LEN, dropout_keep_prob)
        print("doclen %s, voclen %s\n"%(inputs.DOC_LEN, inputs.VOC_LEN))
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
            eval_once(saver, summary_writer, top_k_op, summary_op, dropout_keep_prob)
            if FLAGS.run_once:
                print("eval only once, stope eval")
                break
            print("sleep for {} seconds".format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

    #CWD = os.getcwd()
    #FLAGS.train_dir = os.path.join(CWD, FLAGS.train_dir)
    # glogbal parameters
    # ===============================
    global CHECKPOINT_DIR, EVAL_DIR
    CHECKPOINT_DIR = os.path.abspath(os.path.join(FLAGS.train_dir, FLAGS.checkpoint_dir))
    EVAL_DIR = os.path.abspath(os.path.join(FLAGS.train_dir, "eval-" + str(int(time.time()))))
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("end")

    if tf.gfile.Exists(CHECKPOINT_DIR):
        dataset = os.path.basename(FLAGS.checkpoint_dir).split('.')[0]
            # if not model.initial_dataset_info(dataset):
        #     return

        if dataset == "rotten":
            model.NUM_CLASSES = 2
            model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8530
            model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2132
            inputs.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8530
            inputs.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2132
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
            model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 25000
            model.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 25000
            inputs.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 25000
            inputs.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 25000
        else:
            print("wrong dataset")

        print(model.NUM_CLASSES)
        if tf.gfile.Exists(EVAL_DIR):
            tf.gfile.DeleteRecursively(EVAL_DIR)
        tf.gfile.MakeDirs(EVAL_DIR)
        evaluate()
        #evaluate2()
    else:
        print("error: cannot find checkpoints directory!")


if __name__ == '__main__':
    tf.app.run()

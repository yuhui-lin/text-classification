from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow as tf
# from cnn_character import model
import input

# Parameters
# ==================================================
FLAGS = tf.app.flags.FLAGS

# Model Hyperparameters
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5",
                           "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 128,
                            "Number of filters per filter size (default: 128)")
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
FLAGS.outputs_dir

# ==================================================
# Output directory for checkpoints and summaries

# functions
# ==================================================



def main(argv=None):
    print("start of main")
    print("\nParameters:")
    # weird bug: add this to enable print all Parameters.
    FLAGS.outputs_dir
    for attr, value in sorted(FLAGS.__flags.iteritems()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("\n end of main")


if __name__ == '__main__':
    tf.app.run()

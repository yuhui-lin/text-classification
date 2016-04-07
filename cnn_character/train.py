from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import helper
import data_helper


# Parameters
# ==================================================

# Model Hyperparameters
tf.app.flags.DEFINE_integer(
	"embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5",
						   "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer(
	"num_filters", 128, "Number of filters per filter size (default: 128)")
tf.app.flags.DEFINE_integer(
	"input_length", 1024, "number of characters in each input sequences (default: 1024)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
						  "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0,
						  "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.app.flags.DEFINE_integer(
	"minibatch_size",
	64,
	"mini Batch Size (default: 64)")
tf.app.flags.DEFINE_integer(
	"num_epochs", 200, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer(
	"evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100,
							"Save model after this many steps (default: 100)")
# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement",
							True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement",
							False, "Log placement of ops on devices")
tf.app.flags.DEFINE_string('datasets_dir', 'data/',
						   """Path to the text classification data directory.""")
tf.app.flags.DEFINE_string("dataset", "rotten",
						   """dataset used to train the neural network: rotten, ag, newsgroups, imdb. (default: rotten_tomato)""")


FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
# weird bug: add this to enable print all Parameters.
FLAGS.minibatch_size
for attr, value in sorted(FLAGS.__flags.iteritems()):
	print("{}={}".format(attr.upper(), value))
print("")


def main(argv=None):
	print("start of main")

	# load datasets
	x_train, y_train, x_test, y_test = helper.load_data(FLAGS.dataset, FLAGS.datasets_dir, 10)
	print ("\ninput sequence sample:\n",x_train[0][0])
	print ("input labels sample:\n",y_train[:10])

	print ("\n end of main")



if __name__ == '__main__':
	tf.app.run()

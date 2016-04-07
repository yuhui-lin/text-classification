import numpy as np
import data_helper


def load_data(dataset, data_dir, input_length):
	"""
	Loads and preprocessed data for the certain dataset.
	Returns input vectors, labels
	"""
	# Load and preprocess data
	sequences, labels, sequences_test, labels_test = data_helper.load_data_and_labels(dataset, data_dir)

	print("\npreprocessing train set:")
	sequences_aligned = data_helper.align_sequences(
		sequences, max_length=input_length)
	sequences_quantized = data_helper.quantize_sequences(
		sequences_aligned, data_helper.alphabet)
	x_train = np.array(sequences_quantized)
	y_train = np.array(labels)

	print("\npreprocessing test set:")
	sequences_aligned_test = data_helper.align_sequences(
		sequences_test, max_length=input_length)
	sequences_quantized_test = data_helper.quantize_sequences(
		sequences_aligned_test, data_helper.alphabet)
	x_test = np.array(sequences_quantized_test)
	y_test = np.array(labels_test)

	return [x_train, y_train, x_test, y_test]

#TODO List
* distinguaish between uppercase and lowercase ??
* split train/test set evenly for each categories ??
* write a seperate program to convert all datasets into TFRecord or cvs. Perhaps upload these files to cloud. TF model can read those formats directly.
* TFRecord data problem:
	* stored in sparse representation to save hard disk space. Transform TFRecord to one hot vector during training. There would be one more depth!! and no all zero vector.
	* stored in one hot vector format. may take way more space, maybe 71 more???
	* stored in sparse representation. manage to create all zero vector.
	* stored in real sparse representation format with variable length. This would take double space if sequence is long. all zeoro vector.

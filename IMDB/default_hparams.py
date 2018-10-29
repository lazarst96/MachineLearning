import tensorflow as tf 

def HParams():
	return tf.contrib.training.HParams(
		learning_rate = 0.01,
		batch_size = 2,
		num_epochs = 2,
		vocabulary_size = 1001,
		prefetch = 20,

		embedding_size = 64,
		number_cells_reccurent_layers = [32,32,16],
		dense_layers = [(16,tf.nn.relu),(1,tf.nn.sigmoid)],

		cost_function = tf.nn.sigmoid_cross_entropy_with_logits,
		optimizer = tf.train.AdamOptimizer,

		cpu_num_cores = 4,


	)
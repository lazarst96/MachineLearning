import tensorflow as tf 

def HParams():
	return tf.contrib.training.HParams(
		learning_rate = 0.001,
		batch_size = 32,
		num_epochs = 20,
		vocabulary_size = 1001,
		prefetch = 320,
		training_set_size=25000,
		test_set_size=3000,
		min_seqlen = 100,
		max_seqlen= 1500,
		max_len_padding = 100,

		embedding_size = 32,
		number_cells_reccurent_layers = [32,32,16],
		dense_layers = [(16,tf.nn.relu),(1,tf.nn.sigmoid)],

		cost_function = tf.nn.sigmoid_cross_entropy_with_logits,
		optimizer = tf.train.AdamOptimizer,

		cpu_num_cores = 4,


	)
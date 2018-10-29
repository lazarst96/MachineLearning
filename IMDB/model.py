import tensorflow as tf 
import numpy as np 

class Model(object):
	def __init__(self,hparams,flag_pipeline,vocab_table,num_gpus=0,iterator=None,dataset=None):
		self.flag_pipeline = flag_pipeline
		self.iterator = iterator
		self.dataset = dataset
		self.num_gpus = num_gpus
		self.batch_size = hparams.batch_size
		self.num_epochs = hparams.num_epochs
		self._build_graph(hparams)

	def _build_graph(self, hparams):
		tf.reset_default_graph()
		self.inputs, self.labels = self._inputs_labels(hparams)
		embedded = self._build_embedding(self.inputs,hparams)

		rnn_outputs, rnn_state = self._build_rnn(embedded,hparams)

		output, logits = self._build_fc_network(rnn_outputs[-1],hparams)

		self.cost = tf.reduce_sum(hparams.cost_function(labels=tf.cast(self.labels,tf.float32), logits=logits))
		self.optimizer = hparams.optimizer(hparams.learning_rate).minimize(self.cost)

	def _inputs_labels(self,hparams):
		if self.flag_pipeline == 1:
			return self.iterator.input, self.iterator.labels
		elif self.flag_pipeline == 2:
			return self.dataset.placeholders()
		elif self.flag_pipeline==3:
			return 1
	def _build_embedding(self, input_words, hparams):
		word_embeddings = tf.get_variable("word_embeddings",[hparams.vocabulary_size, hparams.embedding_size])
		embedded = tf.nn.embedding_lookup(word_embeddings, input_words)
		return embedded
	
	def _build_rnn(self, inputs, hparams):
		cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in hparams.number_cells_reccurent_layers]
		rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

		return tf.nn.dynamic_rnn(rnn_cell,inputs,dtype=tf.float32)
	def _build_fc_network(self, inputs, hparams):
		outputs = inputs 
		logits = inputs
		for layer in hparams.dense_layers:
			logits = tf.layers.dense(outputs,layer[0])
			outputs = layer[1](logits)
		return outputs, logits

	def fit(self,helper):
		with tf.Session() as sess:
			self.iterator.training_mode(sess)
			sess.run(helper.init())
			for i in range(self.num_epochs):
				while True:
					try:
						sess.run([
							self.optimizer, 
							self.accuracy, 
							self.cost, 
							self.prediction,
							self.merged
						])
					except tf.errors.OutOfRangeError:
						break
	#def _fit_tfdata(self,session):

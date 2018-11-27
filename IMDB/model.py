import tensorflow as tf 
import numpy as np 
import time
import datetime
from iterator import *

class Model(object):
	def __init__(self,hparams,flag_pipeline,training_dataset,test_dataset,summary_dir):
		self.flag_pipeline = flag_pipeline
		self.iterator = Iterator(training_set=training_dataset.get(),test_set=test_dataset.get(),flag_pipeline=flag_pipeline)
		self.num_epochs = hparams.num_epochs
		self.writer = tf.summary.FileWriter(summary_dir)
		self.num_iter_epoch = hparams.training_set_size//hparams.batch_size + (hparams.training_set_size%hparams.batch_size>0) 
		self._build_graph(hparams)

	@staticmethod
	def _seqlen(seq):
		return tf.reshape(tf.reduce_sum(tf.cast(dtype=tf.int32, x=tf.cast(dtype=tf.bool, x=seq)),axis=1),shape=[tf.shape(seq)[0]])
	def _build_graph(self, hparams):
		self.inputs, self.labels = self.iterator.get_next()
		self.seqlen = self._seqlen(self.inputs)
		self.maxseqlen = tf.reduce_max(self.seqlen)
		self.batch_size = tf.shape(self.inputs)[0]
		with tf.name_scope("Embedded"):
			embedded = self._build_embedding(self.inputs,hparams)
		with tf.name_scope("RNN"):
			rnn_outputs, rnn_state = self._build_rnn(embedded,hparams)
		with tf.name_scope("FullyConnected"):
			output, logits = self._build_fc_network(rnn_outputs,hparams)
			output = tf.round(output)
			self.output = output = tf.cast(output,dtype=tf.int32)

		with tf.name_scope("Cost"):
			self.cost = tf.reduce_sum(hparams.cost_function(labels=tf.cast(self.labels,tf.float32), logits=logits))
		with tf.name_scope("Accuracy"):
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.reduce_sum(self.output,1), tf.reduce_sum(self.labels,1)), tf.float32))
		with tf.name_scope("Optimizer"):	
			self.optimizer = hparams.optimizer(hparams.learning_rate).minimize(self.cost)
		tf.summary.scalar('cost', self.cost/hparams.batch_size)
		tf.summary.scalar('accuracy', self.accuracy)
		self.merged = tf.summary.merge_all()

		

	def _build_embedding(self, input_words, hparams):
		word_embeddings = tf.get_variable("word_embeddings",[hparams.vocabulary_size, hparams.embedding_size])
		embedded = tf.nn.embedding_lookup(word_embeddings, input_words)
		return embedded
	
	def _build_rnn(self, inputs, hparams):
		cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in hparams.number_cells_reccurent_layers]
		rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
		
		full_output, state = tf.nn.dynamic_rnn(cell=rnn_cell,inputs=inputs,dtype=tf.float32,sequence_length=self.seqlen)

		index = tf.range(0, self.batch_size) * self.maxseqlen + (self.seqlen - 1)
		out_size = int(full_output.get_shape()[2])
		full_output = tf.reshape(full_output, [-1, out_size])

		#Za svaku sekvencu u batch-u zadnji izlaz koji nije nula
		outputs =tf.gather(params=full_output, indices=index)
		return outputs, state
	def _build_fc_network(self, inputs, hparams):
		outputs = inputs 
		logits = inputs
		for layer in hparams.dense_layers:
			logits = tf.layers.dense(outputs,layer[0])
			outputs = layer[1](logits)
		return outputs, logits

	def fit(self,verbose=True):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			self.writer.add_graph(sess.graph)
			sum_cost = sum_accuracy = 0
			start_time = time.time()
			print("--Start training:")
			for i in range(1,self.num_epochs+1):
				self.iterator.test_mode(sess)
				accuracy,cost,summary=sess.run([
					self.accuracy, 
					self.cost,
					self.merged
				])
				print("  Cost {:.5}, Accuracy {:.4}%".format(cost,accuracy*100))
				self.writer.add_summary(summary,i)
				self.iterator.training_mode(sess)
				print("-Start Epoch {}:".format(i))
				it=1
				prev_it=0
				while True:
					try:
						_,accuracy,cost,summary=sess.run([
							self.optimizer, 
							self.accuracy, 
							self.cost,
							self.merged
						])
						sum_cost +=cost
						sum_accuracy +=accuracy
						#self.writer.add_summary(summary,i)
						if(verbose and (it%100==0 or it==self.num_iter_epoch)):
							avg_cost = sum_cost/(it-prev_it)
							avg_accuracy = sum_accuracy/(it-prev_it)
							sum_cost = sum_accuracy = 0
							prev_it = it
							print("{}/{} - cost: {:.5}, accuracy: {:.4}%".format(it,self.num_iter_epoch,avg_cost,avg_accuracy*100))
						it+=1
					except tf.errors.OutOfRangeError:
						print("-End Epoch {}.".format(i))
						break
			print("--End training.")
			training_time = int(time.time()-start_time)
			print("Training time: {}".format(str(datetime.timedelta(seconds=training_time))))
			print("Average per epoch: {}".format(str(datetime.timedelta(seconds=training_time//self.num_epochs))))
			self.iterator.test_mode(sess)
			accuracy,cost,summary=sess.run([
				self.accuracy, 
				self.cost,
				self.merged
			])
			self.writer.add_summary(summary,self.num_epochs+1)
			print("Cost {:.5}, Accuracy {:.4}%".format(cost,accuracy*100))

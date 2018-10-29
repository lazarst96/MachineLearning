import random
import tensorflow as tf

class CustomDataset(object):
	def __init__(self, filenames, labels, hparams, map_func):
		self.inputs = list.copy(filenames)
		self.labels = list.copy(labels)
		self.batch_size = hparams.batch_size
		self.num_epochs = hparams.num_epochs

		z = list(zip(self.inputs, self.labels))
		random.shuffle(z)
		self.inputs = list([*zip(*z)][0])
		self.labels = list([*zip(*z)][1])

		self.dataset_size = len(self.inputs)
		self.num_batches = int(self.dataset_size/self.batch_size)+int(self.dataset_size%self.batch_size>0)
		self.num_iterations= self.num_batches*self.num_epochs
		self.it = 0
		
		self.map_func = map_func

	def get_next(self):
		if(self.it>=self.num_iterations):
			raise Exception("End of Dataset")
		start = (self.it%self.num_batches)*self.batch_size
		end = min(start + self.batch_size,self.dataset_size)
		inputs = self.inputs[start:end]
		labels = self.labels[start:end]
		self.it+=1
		return inputs, labels
	def placeholders(self):
		inputs = tf.placeholders(tf.string,[None])
		labels = tf.placeholders(tf.int32,[None,1])
		inputs = tf.map_fn(self.map_func,inputs)
		return inputs, labels
	def init(self,session):
		self.helper.init(session)

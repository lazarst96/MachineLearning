import tensorflow as tf 
from helper import *

class Dataset(object):
	def __init__(self,filenames, labels, helper, hparams, test=False):
		self.inputs = tf.data.TextLineDataset(filenames)
		self.labels = tf.data.Dataset.from_tensor_slices(labels)
		self.helper = helper
		self.dataset = tf.data.Dataset.zip((self.inputs,self.labels))
		self.dataset = self.dataset.map(helper.preprocessing_tfdata, num_parallel_calls=hparams.cpu_num_cores)
		if(test):
			self.dataset = self.dataset.padded_batch(hparams.test_set_size,padded_shapes=([None],1))
		else:
			self.dataset = self.dataset.prefetch(hparams.prefetch)
			self.dataset = self.dataset.padded_batch(hparams.batch_size,padded_shapes=([None],1))
	def get(self):
		return self.dataset
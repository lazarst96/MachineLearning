import tensorflow as tf 
from helper import *

class Dataset(object):
	def __init__(self,filenames, labels, helper, hparams):
		self.inputs = tf.data.TextLineDataset(filenames)
		self.labels = tf.data.Dataset.from_tensor_slices(labels)
		#self.inputs = self.inputs.map(helper.preprocessing_tfdata, num_parallel_calls=hparams.cpu_num_cores)
		self.dataset = tf.data.Dataset.zip((self.inputs,self.labels)).map(helper.preprocessing_tfdata, num_parallel_calls=hparams.cpu_num_cores).shuffle(50000).repeat(hparams.num_epochs).prefetch(hparams.prefetch)
		#padded_batch(hparams.batch_size,padded_shapes=(tf.TensorShape([None]),tf.TensorShape([None])),padding_values=(-1,0))
		self.helper = helper
		self.iterator = self.dataset.make_initializable_iterator()
	def get_iterator(self):
		inputs = self.iterator.get_next()[0]
		labels = self.iterator.get_next()[1]
		return inputs, labels
	def init(self,session):
		self.helper.init(session)
		session.run(self.iterator.initializer)
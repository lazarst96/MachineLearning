import tensorflow as tf 
from helper import *

class Dataset(object):
	def __init__(self,filenames,labels,helper,hparams,test=False):
		self.inputs = tf.data.TextLineDataset(filenames)
		self.labels = tf.data.Dataset.from_tensor_slices(labels)
		self.helper = helper
		self.dataset = tf.data.Dataset.zip((self.inputs,self.labels))
		self.dataset = self.dataset.map(helper.preprocessing_tfdata, num_parallel_calls=hparams.cpu_num_cores)
		if(test):
			self.dataset = self.dataset.padded_batch(hparams.test_set_size,padded_shapes=([None],1))
		else:
			batch_sizes, bucket_boundaries = self._batch_classes(hparams.batch_size, hparams.min_seqlen, hparams.max_seqlen, hparams.max_len_padding)
			self.dataset = self.dataset.prefetch(hparams.prefetch)
			self.dataset = self.dataset.apply(tf.data.experimental.bucket_by_sequence_length(element_length_func=self._element_length,bucket_batch_sizes=batch_sizes,bucket_boundaries=bucket_boundaries,padded_shapes=([None],1)))
	def get(self):
		return self.dataset
	@staticmethod
	def _element_length(x,y):
		return tf.shape(x)[0]
	@staticmethod
	def _batch_classes(batch_size,min_seqlen,max_seqlen,max_len_padding):
		num_class = (max_seqlen - min_seqlen)//max_len_padding + ((max_seqlen - min_seqlen)%max_len_padding > 0)
		batch_sizes = [batch_size]
		bucket_boundaries = []
		for i in range(1,num_class+1):
			batch_sizes+=[batch_size]
			bucket_boundaries += [min_seqlen+i*max_len_padding]
		return batch_sizes, bucket_boundaries
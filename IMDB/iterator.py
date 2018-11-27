import tensorflow as tf 
import numpy as np 

class Iterator(object):
	def __init__(self,training_set,test_set,flag_pipeline):
		self.training_set = training_set
		self.test_set = test_set
		self.iterator = tf.data.Iterator.from_structure(
			training_set.output_types,
			training_set.output_shapes)
		self.init = None
	def training_mode(self,session):
		session.run(self.iterator.make_initializer(self.training_set))
	def test_mode(self,session):
		session.run(self.iterator.make_initializer(self.test_set))
	def get_next(self):
		return self.iterator.get_next()
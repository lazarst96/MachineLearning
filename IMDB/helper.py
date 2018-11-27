import tensorflow as tf
import nltk
import numpy as np

class Helper(object):
	def __init__(self,dictionary_path):
		dfile = open(dictionary_path,encoding="utf8")
		keys = []
		values = []
		self.dict ={} 
		for line in dfile:
			l = line.split()
			keys.append(l[0])
			values.append(int(l[1]))
			self.dict[l[0]] = int(l[1])

	def preprocessing_feeddict(self,filename):
		return tf.py_func(self._py_preprocessing,[inputs],[tf.int32])[0],labels
	def _py_preprocessing(self,string):
		def _transform(val):
			if(val==None):
				return -1
			else:
				return val
		
		string = string.decode("utf-8")
		l = nltk.word_tokenize(string)
		l = [_transform(self.dict.get(i)) for i in l]
		a = np.array(l)
		b = (a>=0)
		return a[b]+1
	def preprocessing_tfdata(self,inputs, labels):
		return tf.py_func(self._py_preprocessing,[inputs],[tf.int32])[0],labels

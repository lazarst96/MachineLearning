import tensorflow as tf

class Helper(object):
	def __init__(self,dictionary_path):
		dfile = open(dictionary_path,encoding="utf8")
		keys = []
		values = []
		for line in dfile:
			l = line.split()
			keys.append(l[0])
			values.append(int(l[1]))
		self.table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)

	def preprocessing_feeddict(self,filename):
		content = tf.read_file(filename)
		words = tf.string_split([content],delimiter=" '").values
		return self.table.lookup(words)+1
	def preprocessing_tfdata(self,inputs, labels):
		words = tf.string_split([inputs],delimiter=" '").values
		return self.table.lookup(words)+1,labels
	def init(self,session):
		session.run(self.table.init)

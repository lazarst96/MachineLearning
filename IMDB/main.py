from model import *
from default_hparams import *
from dataset_custom import *
from dataset import *

'''filenames = ["file1.txt","file2.txt","file3.txt","file4.txt"]
labels = [1,2,3,4]

dataset = CustomDataset(filenames, labels, HParams(), None)
while True:
	try:
		print(dataset.get_next())
	except Exception:
		print("---End of dataset---")
		break'''
labels = [1,1,1,0,0]
filenames = ["./aclImdb/train/pos/5_10.txt","./aclImdb/train/pos/5_10.txt","./aclImdb/train/pos/1_7.txt","./aclImdb/train/pos/2_9.txt","./aclImdb/train/neg/0_3.txt","./aclImdb/train/neg/1_1.txt"]
with tf.Session() as sess:
	helper = Helper("./aclImdb/dictionary.txt")
	dataset = Dataset(filenames, labels, helper, HParams())
	dataset.init(sess)
	inp, lab = dataset.get_iterator()
	print(sess.run([inp,lab]))

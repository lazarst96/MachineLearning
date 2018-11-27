from model import *
from hparams import *
from dataset import *
#from dataset_group_by_seqlen import *
import os
import random

train_path = "./aclImdb/train/"
test_path = "./aclImdb/test/"

train_filenames = [train_path+"pos/"+i for i in (os.listdir(train_path+"pos"))] + [train_path+"neg/"+i for i in (os.listdir(train_path+"neg"))]
test_filenames = [test_path+"pos/"+i for i in (os.listdir(test_path+"pos"))] + [test_path+"neg/"+i for i in (os.listdir(test_path+"neg"))]
train_labels = [[1] for _ in range(12500)] + [[0] for _ in range(12500)]
test_labels = [[1] for _ in range(12500)] + [[0] for _ in range(12500)]

start_state = random.getstate()
random.shuffle(train_filenames)
random.setstate(start_state)
random.shuffle(train_labels)

start_state = random.getstate()
random.shuffle(test_filenames)
random.setstate(start_state)
random.shuffle(test_labels)


hparams = HParams()
helper = Helper("./aclImdb/dictionary.txt")
training_dataset = Dataset(train_filenames, train_labels, helper, hparams)
test_dataset = Dataset(test_filenames[:3000], test_labels[:3000], helper, hparams, test=True)
model = Model(hparams=hparams,flag_pipeline=1,training_dataset=training_dataset,test_dataset=test_dataset,summary_dir="tensorboard/test/")
model.fit(verbose=True)

from mpl_toolkits.mplot3d import Axes3D
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import math
import random
from mlxtend.data import loadlocal_mnist

class NeuralNetwork(object):
	def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, activation_func_input= "sigmoid",activation_func_hidden= "identity",activation_func_output="identity"):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.learning_rate = learning_rate

		self.in_hid_weights = np.matrix(np.random.randn(input_size, hidden_size))
		self.hid_out_weights = np.matrix(np.random.randn(hidden_size, output_size))

		self.prev_in_hid_grad = np.matrix(np.zeros((input_size, hidden_size,)))
		self.prev_hid_out_grad = np.matrix(np.zeros((hidden_size, output_size)))
		self.prev_hid_out_bias = np.matrix(np.zeros(output_size))
		self.gama = 0.9

		self.hid_out_bias = np.matrix(np.random.randn(output_size))

		activation_func_output_split=activation_func_output.split()
		activation_func_output=activation_func_output_split[0]
		#print(activation_func_output)
		if len(activation_func_output_split) > 1:
			self.label=float(activation_func_output_split[1])
		else:
			self.label=0

		self.activation_func_input=NeuralNetwork._activacion_funct(activation_func_input)
		self.derivat_func_input=NeuralNetwork._derivat_funct(activation_func_input)
		self.activation_func_hidden=NeuralNetwork._activacion_funct(activation_func_hidden)
		self.derivat_func_hidden=NeuralNetwork._derivat_funct(activation_func_hidden)
		self.activation_func_output=self._activacion_funct_out(activation_func_output)
		
	def _sigmoid(x):
		return 1/(1+math.exp(-x))
	def _dsigmoid(x):
		return NeuralNetwork._sigmoid(x)*(1-NeuralNetwork._sigmoid(x))
	def _tanh(x):
		return math.tanh(x)
	def _dtanh(x):
		return 1/(math.cosh(x)**2)
	def _identity(x):
		return x
	def _didentity(x):
		return 1
	def _signum(self,x):
		if x<self.label:
			return 0
		else:
			return 1
	def _signum_array(self,array):
		return np.vectorize(self._signum)(array)
	def _softmax_array(slef,array):
		exps=np.exp(array-np.max(array))
		return exps / np.sum(exps)
	def _identity_array(self,array):
		return array
	def _activacion_funct(string):
		return{
			"tanh": NeuralNetwork._tanh,
			"identity":NeuralNetwork._identity,
		}.get(string,NeuralNetwork._sigmoid)
	def _activacion_funct_out(self,string):
		return{
			"identity":NeuralNetwork._identity_array,
			"signum":self._signum_array,
			"softmax":self._softmax_array
		}.get(string,NeuralNetwork._identity_array)
	def _derivat_funct(string):
		return{
			"tanh": NeuralNetwork._dtanh,
			"identity":NeuralNetwork._didentity,
		}.get(string,NeuralNetwork._dsigmoid)
	def train(self, input_array, target):
		if len(input_array)!=self.input_size:
			raise Exception("Wrong size of input array")

		input_array= np.matrix(input_array)
		target = np.matrix(target)

		hidden_layer_input = input_array.dot(self.in_hid_weights)
		#print(hidden_layer_input)
		hidden_layer_output = np.vectorize(self.activation_func_input)(hidden_layer_input)

		output_layer_input = np.add(hidden_layer_output.dot(self.hid_out_weights),self.hid_out_bias)
		output_layer_output = np.vectorize(self.activation_func_hidden)(output_layer_input)
		output_layer_output = self.activation_func_output(output_layer_output)

		errors = target - output_layer_output
		
		gradient_outputs = np.vectorize(self.derivat_func_hidden)(output_layer_input)
		gradient_outputs=np.multiply(gradient_outputs,errors)

		delta_w_output = np.transpose(hidden_layer_output).dot(gradient_outputs)
		
		gradient_hidden = np.dot(gradient_outputs,np.transpose(self.hid_out_weights))
		#print(gradient_hidden)
		gradient_hidden = np.multiply(gradient_hidden,np.vectorize(self.derivat_func_input)(hidden_layer_input))

		delta_w_hidden = np.dot(np.transpose(input_array),gradient_hidden)
		
		
		self.prev_in_hid_grad = np.multiply(self.prev_in_hid_grad,self.gama) - np.multiply(delta_w_hidden,self.learning_rate)
		self.prev_hid_out_grad = np.multiply(self.prev_hid_out_grad,self.gama) - np.multiply(delta_w_output,self.learning_rate)
		self.prev_hid_out_bias = np.multiply(self.prev_hid_out_bias,self.gama) - np.multiply(gradient_outputs,self.learning_rate)
		
		self.hid_out_weights = self.hid_out_weights - self.prev_hid_out_grad
		
		self.in_hid_weights = self.in_hid_weights - self.prev_in_hid_grad
		self.hid_out_bias = self.hid_out_bias - self.prev_hid_out_bias




	def getResult(self,input_array):
		if len(input_array)!=self.input_size:
			raise Exception("Wrong size of input array")

		input_array= np.matrix(input_array)

		hidden_layer_input = input_array.dot(self.in_hid_weights)
		hidden_layer_output = np.vectorize(self.activation_func_input)(hidden_layer_input)

		output_layer_input = np.add(hidden_layer_output.dot(self.hid_out_weights),self.hid_out_bias)
		output_layer_output = np.vectorize(self.activation_func_hidden)(output_layer_input)

		return self.activation_func_output(output_layer_output)
	def printWeightsInFile(self,fileName):
		open(fileName,"w").close()
		file=open(fileName,"ab")
		np.savetxt(file, self.in_hid_weights, delimiter=' ')
		np.savetxt(file, self.hid_out_weights, delimiter=' ')
		np.savetxt(file, self.hid_out_bias, delimiter=' ')
		np.savetxt(file, self.prev_in_hid_grad, delimiter=' ')
		np.savetxt(file, self.prev_hid_out_grad, delimiter=' ')
		np.savetxt(file, self.prev_hid_out_bias, delimiter=' ')
		file.close()
	def loadWeightsFromFile(self,fileName):
		file = open(fileName,"r")
		l=[]
		for i in range(0,self.input_size):
			l.append(np.vectorize(float)(file.readline().split()))
		self.in_hid_weights = np.matrix(l)

		l=[]
		for i in range(0,self.hidden_size):
			l.append(np.vectorize(float)(file.readline().split()))
		self.hid_out_weights = np.matrix(l)

		self.hid_out_bias = np.matrix(np.vectorize(float)(file.readline().split()))

		l=[]
		for i in range(0,self.input_size):
			l.append(np.vectorize(float)(file.readline().split()))
		self.prev_in_hid_grad=np.matrix(l)

		l=[]
		for i in range(0,self.hidden_size):
			l.append(np.vectorize(float)(file.readline().split()))
		self.prev_hid_out_grad=np.matrix(l)

		self.prev_hid_out_bias = np.matrix(np.vectorize(float)(file.readline().split()))

		file.close()


#end-Neural-Network
def func(x):
	ar = np.zeros(10)
	ar[x]=1
	return ar
#end-func


nn = NeuralNetwork(784,100,10,0.1,"sigmoid","sigmoid","softmax")
X, y = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')

nn.loadWeightsFromFile("weights.txt")
#nn.printWeightsInFile("weights.txt")
wrongAnsX=[]
wrongAnsY=[]
correct = np.zeros(10)
wrong = np.zeros((10,10))
cnt = np.zeros(10)
tX, ty = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte', 
        labels_path='t10k-labels.idx1-ubyte')
for i in range(tX.shape[0]):
	rez = np.argmax(nn.getResult(tX[i]/255)) 
	cnt[ty[i]]=cnt[ty[i]]+1
	
	if rez != ty[i]:
		wrong[ty[i]][rez]=wrong[ty[i]][rez]+1
		wrongAnsX.append(tX[i])
		wrongAnsY.append(rez)
		
	else:
		correct[ty[i]] = correct[ty[i]]+1

#print("Precision = {0:.2f}%".format(correct*100/tX.shape[0]))

height=np.divide(correct,cnt)*100

bars = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.show()

for i in range(10):
	wrong[i]= wrong[i]*100 / (cnt[i])
df = pd.DataFrame(wrong, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sns.heatmap(df, cmap="Greens")
plt.show()

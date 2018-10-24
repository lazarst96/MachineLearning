from ActivationFunction import *
from Optimizer import *
from LossFunction import *
from LSTMLayer import *
from GRULayer import *
from VanillaRNNLayer import *
from DenseLayer import *


class RecurrentNeuralNetwork(object):
	def __init__(self,input_size, batch_size, optimizer : Optimizer,lossfunction : LossFunction):
		self.optimizer = optimizer
		self.lossfunction = lossfunction
		self.layers = []
		self.input_size = input_size
		self.batch_size = batch_size

	#ocekuje liste kao parametre za input dimeznije (epoch_size x batch_size x input_size)
	def fit(self, input_epoch, target_epoch):
		outputs=self.feed_forward(input_epoch)
		targets = np.array(target_epoch)
		targets = np.reshape(targets,[targets.shape[0],targets.shape[1],1,targets.shape[2]])
		gradients = self._backward(outputs, targets)
		self._optimize_weights(gradients)
		loss = self.lossfunction.function(outputs,targets)
		return loss

	#ocekuje listu kao parametar
	def feed_forward(self,input_epoch):
		for i in range(len(self.layers)):
			self.layers[i].reset_state()

		output_seq = []
		for input_batch in input_epoch:
			inp = np.array(input_batch)
			inp = np.reshape(inp,[inp.shape[0],1,inp.shape[1]])
			for i in range(len(self.layers)):
				inp = self.layers[i].forward(inp)
			output_seq.append(inp)

		return np.array(output_seq)
	#ocekuje np.array kao paremtre dimenzija (epoch_size x batch_size x 1 x output_size)
	def _backward(self,output_epoch,target_epoch):
		gradients = []
		start = len(self.layers)-1
		for_prev = self.lossfunction.derivate(output_epoch,target_epoch)

		if isinstance(self.lossfunction, CrossEntropyWithSoftmax) or isinstance(self.lossfunction, CrossEntropyWithSigmoid):
			start = len(self.layers)-2
			gradient, for_prev = self.layers[-1].backward_for_cross_entropy(for_prev)
			gradients.append(gradient)

		
		for i in range(start,-1,-1):
			
			gradient, for_prev = self.layers[i].backward(for_prev)
			
			gradients.append(gradient)

		return gradients

	def _optimize_weights(self,gradients):
		deltas = self.optimizer.delta(gradients)
		for i in range(len(self.layers)-1,-1,-1):
			self.layers[i].optimize_weights(deltas[len(self.layers)-i-1])

	def addDenseLayer(self,layer_size,bias,activationfunction : ActivationFunction):
		if len(self.layers)>0:
			layer_input_size = self.layers[-1].size
		else:
			layer_input_size = int(self.input_size)
		self.layers.append(DenseLayer(layer_input_size, layer_size, bias, activationfunction))
		for i in range(len(self.layers)-1):
			self.layers[i].reset()

	def addBasicRnnLayer(self,layer_size,bias,activationfunction : ActivationFunction):
		if len(self.layers)>0:
			layer_input_size = self.layers[-1].size
		else:
			layer_input_size = int(self.input_size)
		self.layers.append(RnnLayer(layer_input_size, layer_size, self.batch_size, bias, activationfunction))
		for i in range(len(self.layers)-1):
			self.layers[i].reset()

	def addGRULayer(self,layer_size,activationfunction : ActivationFunction):
		if len(self.layers)>0:
			layer_input_size = self.layers[-1].size
		else:
			layer_input_size = int(self.input_size)
		self.layers.append(GRULayer(layer_input_size, layer_size, self.batch_size, activationfunction))
		for i in range(len(self.layers)-1):
			self.layers[i].reset()
	def addLSTMLayer(self,layer_size,activationfunction : ActivationFunction):
		if len(self.layers)>0:
			layer_input_size = self.layers[-1].size
		else:
			layer_input_size = int(self.input_size)
		self.layers.append(LSTMLayer(layer_input_size, layer_size, self.batch_size, bias, activationfunction))
		for i in range(len(self.layers)-1):
			self.layers[i].reset()



model = RecurrentNeuralNetwork(2, 2, Adam(0.001), MSE())
model.addGRULayer(4, ELU())
print(model.feed_forward([[[1,0],[0.3,0.1]],[[0.8,0.1],[0.3,0.4]]]).shape)
model.fit([[[1,0],[0.3,0.1]],[[0.8,0.1],[0.3,0.4]],[[0.8,0.1],[0.3,0.4]]], [[[1],[0]],[[0],[0]],[[0],[1]]])







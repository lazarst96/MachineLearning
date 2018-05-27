from ActivationFunction import *
from Optimizer import *
from LossFunction import *
from Layer import *

class NeuralNetwork(object):
	def __init__(self,input_size, optimizer : Optimizer,lossfunction : LossFunction):
		self.optimizer = optimizer
		self.lossfunction = lossfunction
		self.layers = []
		self.input_size = input_size

			

	#ocekuje liste kao parametre
	def fit(self, input_batch, target_batch):
		outputs=self.feed_forward(input_batch)
		targets = np.array([target_batch])
		gradients = self._backward(outputs, targets)
		self._optimize_weights(gradients)
		loss = self.lossfunction.function(outputs,targets)
		return loss

	#ocekuje listu kao parametar
	def feed_forward(self,input_batch):
		inp = np.array(input_batch)
		inp = np.reshape(inp,[inp.shape[0],1,inp.shape[1]])
		for i in range(len(self.layers)):
			inp = self.layers[i].forward(inp)
		return inp

	def _backward(self,output_batch,target_batch):
		gradients = []
		start = len(self.layers)-1
		for_prev = self.lossfunction.derivate(output_batch,target_batch)

		if isinstance(self.lossfunction, CrossEntropyWithSoftmax) or isinstance(self.lossfunction, CrossEntropyWithSigmoid):
			start = len(self.layers)-2
			gradient, for_prev = self.layers[-1].backward_for_cross_entropy(self.lossfunction.derivate(output_batch, target_batch))
			gradients.append(gradient)

		
		for i in range(start,-1,-1):
			
			gradient, for_prev = self.layers[i].backward(for_prev)
			
			gradients.append(gradient)

		return gradients

	def _optimize_weights(self,gradients):
		deltas = self.optimizer.delta(gradients)
		for i in range(len(self.layers)-1,-1,-1):
			self.layers[i].optimize_weights(deltas[len(self.layers)-i-1])


	def addLayer(self,layer_size,bias,activationfunction : ActivationFunction):
		if len(self.layers)>0:
			layer_input_size = self.layers[-1].size
		else:
			layer_input_size = int(self.input_size)
		self.layers.append(Layer(layer_input_size, layer_size, bias, activationfunction))
		for i in range(len(self.layers)-1):
			self.layers[i].reset()

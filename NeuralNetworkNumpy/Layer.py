from ActivationFunction import *
import numpy as np
import copy

class Layer(object):
	def __init__(self,input_size,size,bias,activationfunction : ActivationFunction):
		self.size = size
		self.input_size = input_size
		self.bias = bias
		if bias:
			self.input_size = self.input_size + 1
		self.activation = activationfunction
		self.weights = np.random.randn(self.input_size, self.size)
		self.input_batch = None
		self.potentials = None

	def forward(self,input_batch):
		self.input_batch = np.copy(input_batch)
		if self.bias:
			self.input_batch=np.dstack((input_batch,np.ones( (input_batch.shape[0],input_batch.shape[1],1) )))

		self.potentials = np.tensordot(self.input_batch,self.weights,1)
		return self.activation.function(self.potentials)

	#sledeci gradient misli se u forward smeru
	def backward(self,from_next):
		gradient = np.multiply(from_next,self.activation.derivate(self.potentials))

		if self.bias:
			weights=np.delete(self.weights,self.weights.shape[0]-1,0)
		else:
			weights = self.weights

		for_prev = np.tensordot(gradient,np.transpose(weights),1)
		gradient = np.multiply(np.transpose(self.input_batch,[0,2,1]), gradient)
		return np.sum(gradient,0), for_prev


	def backward_for_cross_entropy(self, from_next):
		gradient = from_next
		if self.bias:
			weights=np.delete(self.weights,self.weights.shape[0]-1,0)
		else:
			weights = self.weights

		for_prev = np.tensordot(gradient,np.transpose(weights),1)
		gradient = np.multiply(np.transpose(self.input_batch,[0,2,1]), gradient)
		return np.sum(gradient,0), for_prev

	def reset(self):
		self.weights = np.random.randn(self.input_size, self.size)

	def optimize_weights(self,delta):
		self.weights = self.weights - delta
		


import math
from abc import ABC, abstractmethod
import numpy as np
class ActivationFunction(ABC):
	@abstractmethod
	def function(self,x : np.array):
		pass
	@abstractmethod
	def derivate(self,x : np.array):
		pass
	

class Sigmoid(ActivationFunction):
	def function(self,x : np.array):
		return 1/(1+np.exp(-x))
	def derivate(self,x : np.array):
		return self.function(x)*(1-self.function(x))
	
class Tanh(ActivationFunction):
	def function(self,x : np.array):
		return np.tanh(x)
	def derivate(self,x : np.array):
		return 1/(np.cosh(x)**2)
	
class Identity(ActivationFunction):
	def function(self,x : np.array):
		return x
	def derivate(self,x : np.array):
		return np.ones(x.shape)
	
class Softmax(ActivationFunction):
	def __init__(self):
		self.sigmoid = Sigmoid()
	def function(self,x : np.array):
		exps=np.exp(x-np.max(x))
		return exps / np.sum(exps)
	def derivate(self,x : np.array):
		xx = self.sigmoid.function(x)
		return np.multiply((self.function(xx)-1),self.function(xx))*(-1)

class ReLU(ActivationFunction):
	def function(self,x : np.array):
		return np.maximum(a,0)
	def derivate(self,x : np.array):
		return np.sign(np.maximum(a,0))

class ELU(ActivationFunction):
	def function(self,x : np.array):
		return (np.exp(x)-1)*np.sign(np.maximum(-x,0))+x*np.sign(np.maximum(x,0))
	def derivate(self,x : np.array):
		return np.exp(x)*np.sign(np.maximum(-x,0))+np.sign(np.maximum(x,0))
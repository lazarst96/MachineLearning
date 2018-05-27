import numpy as np
import math
from abc import ABC, abstractmethod
import copy

class Optimizer(ABC):
	@abstractmethod
	def delta(self,gradients):
		pass

class GradientDescent(Optimizer):
	def __init__(self,learning_rate):
		self.learning_rate = learning_rate
	def delta(self,gradients):
		grad = np.matrix(gradients)
		return (grad*self.learning_rate).tolist()[0]

class Momentum(Optimizer):
	def __init__(self,learning_rate,decay=0.9):
		self.learning_rate = learning_rate
		self.decay = decay
		self.flag = True
		self.prev_grad = []
	def delta(self,gradients):
		if self.flag:
			self.flag = False
			for i in gradients:
				self.prev_grad.append(np.matrix(np.zeros((i.shape[0],i.shape[1]))))
		for i in range(len(gradients)):
			self.prev_grad[i] = self.prev_grad[i]*self.decay + self.learning_rate * gradients[i]
		return copy.deepcopy(self.prev_grad)

class AdaGrad(Optimizer):
	def __init__(self,learning_rate):
		self.learning_rate = learning_rate
		self.eps = math.exp(-8) 
		self.sum_sq = []
		self.flag = True
	def delta(self,gradients):
		if self.flag:
			self.flag = False
			for i in gradients:
				self.sum_sq.append(np.matrix(np.zeros((i.shape[0],i.shape[1]))))
		delta =gradients
		for i in range(len(gradients)):
			self.sum_sq[i] = self.sum_sq[i] + np.square(gradients[i])
			delta[i] = np.multiply(delta[i],(np.sqrt(np.power(self.sum_sq[i]+self.eps,-1))*self.learning_rate))
		return delta

class Adam(Optimizer):
	def __init__(self,learning_rate,beta1=0.9,beta2=0.99):
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = 10**(-8)
		self.v = []
		self.m = []
		self.t = 1
	def delta(self,gradients):
		if self.t==1:
			for i in gradients:
				self.m.append(np.matrix(np.zeros((i.shape[0],i.shape[1]))))
				self.v.append(np.matrix(np.zeros((i.shape[0],i.shape[1]))))
		delta = gradients
		self.t = self.t+1
		for i in range(len(gradients)):
			self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*gradients[i]
			self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*np.square(gradients[i])
			mt = self.m[i]/(1-self.beta1**self.t)
			vt = self.v[i]/(1-self.beta2**self.t)
			delta[i] = np.multiply(np.power(np.sqrt(vt)+self.eps,-1)*self.learning_rate,mt)
		return delta

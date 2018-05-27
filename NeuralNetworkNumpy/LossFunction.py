import math
import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
	@abstractmethod
	def function(self,output,target):
		pass
	@abstractmethod
	def derivate(self,output,target):
		pass

class MSE(LossFunction):
	def function(self,output,target):
		return np.sum(np.square(target-output))/len(output)
	def derivate(self,output,target):
		return output-target
	
class CrossEntropyWithSoftmax(LossFunction):
	def function(self,output,target):
		return -np.sum(target*np.log(output))
	def derivate(self,output,target):
		return output - target

class CrossEntropyWithSigmoid(LossFunction):
	def function(self,output,target):
		return np.sum(-(target*np.log(output) + (1-target)*np.log(1-output)))
	def derivate(self,output,target):
		return output -target


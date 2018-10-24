from abc import ABC, abstractmethod
from ActivationFunction import *


class Layer(ABC):
	@abstractmethod
	def forward(self, input_batch):
		pass
	@abstractmethod
	def backward(self,from_next):
		pass
	@abstractmethod
	def optimize_weights(self,delta):
		pass
	@abstractmethod
	def reset(self):
		pass
	@abstractmethod
	def reset_state(self):
		pass




		


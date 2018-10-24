from RecurrentLayer import *

class LSTMLayer(RecurrentLayer):
	def __init__(self,input_size,size, batch_size, activationfunction  : ActivationFunction = Tanh()):
		self.size = size
		self.input_size = input_size
		self.bias = bias
		self.batch_size = batch_size
		self.input_size = self.input_size + 1

		self.activation = activationfunction
		self.sigmoid = Sigmoid()
		self.tanh = Tanh()

		#tezine za input gate
		self.wi = np.random.randn(self.size+self.input_size, self.size)
		#tezine za forget gate
		self.wf = np.random.randn(self.size+self.input_size, self.size)
		#tezine za output gate
		self.wo = np.random.randn(self.size+self.input_size, self.size)
		#tezine za memory cell candidate
		self.w = np.random.randn(self.size+self.input_size, self.size)


		self.current_state = np.zeros((self.batch_size, 1, self.size))
		self.current_memory_cell = np.zeros((self.batch_size, 1, self.size))
		
	def forward(self, input_batch):
		input_batch=np.dstack((input_batch,np.ones( (input_batch.shape[0],input_batch.shape[1],1) )))
		state_inp = np.dstack((input_batch, self.current_state))

		input_gate = self.sigmoid.function(np.tensordot(state_inp,self.wi,1))
		forget_gate = self.sigmoid.function(np.tensordot(state_inp,self.wf,1))
		output_gate = self.sigmoid.function(np.tensordot(state_inp,self.wo,1))

		memory_cell_candidate = self.tanh.function(np.tensordot(state_inp,self.w,1))

		self.current_memory_cell = np.multiply(self.current_memory_cell, forget_gate) + np.multiply(memory_cell_candidate, input_gate)

		self.current_state = np.multiply(self.tanh.function(self.current_memory_cell), output_gate)

		return self.current_state

		
	def backward(self,from_next):
		pass
	def optimize_weights(self,delta):
		pass
	def reset(self):
		pass
	def reset_state(self):
		pass
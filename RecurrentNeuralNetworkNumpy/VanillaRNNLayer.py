from RecurrentLayer import *


class RnnLayer(RecurrentLayer):
	def __init__(self,input_size,size, batch_size, bias, activationfunction : ActivationFunction):
		self.size = size
		self.input_size = input_size
		self.bias = bias
		self.batch_size = batch_size
		if bias:
			self.input_size = self.input_size + 1
		self.activation = activationfunction
		self.weights = np.random.randn(self.size+self.input_size, self.size)
		self.current_state = np.zeros((self.batch_size, 1, self.size))
		self.state_inp = []
		self.potentials = []

	def forward(self, input_batch):
		if self.bias:
			input_batch=np.dstack((input_batch,np.ones( (input_batch.shape[0],input_batch.shape[1],1) )))

		#self.input_batches.append(input_batch)

		#spojeno current state i input
		state_inp = np.dstack((input_batch, self.current_state))
		self.state_inp.append(state_inp)
		potential = np.tensordot(state_inp,self.weights,1)

		self.potentials.append(potential)
		self.current_state = self.activation.function(potential)
		return self.current_state

	def reset_state(self):
		self.state_inp = []
		self.potentials = []
		self.current_state = np.zeros((self.batch_size, 1, self.size))

	def backward(self,from_next):
		#tezine sinapsi za ulaz
		if self.bias:
			wu=self.weights[0:self.input_size-1,:]
		else:
			wu=self.weights[0:self.input_size,:]

		#tezine sinapsi za prethodno stanje
		wx = self.weights[self.input_size:,:]

		potentials = np.array(self.potentials)

		#parcijalni izvod aktivnosti po potencijalu
		dak_dpk = self.activation.derivate(potentials)

		#parcijalni izvod potencijala u stanju k po stanju k-1
		dpk_dxk_1 = np.transpose(wx)

		#parcijalni izvod aktivnosti u stanju k po stanju k-1
		dak_dxk_1 = np.tensordot(dak_dpk, dpk_dxk_1, 1)

		#uredjeni parcijalni izvod (suma gresaka od k do broj epoha) po stanju k
		gradient = from_next

		for i in range(from_next.shape[0]-2,-1,-1):
			gradient[i] += np.multiply(gradient[i+1], dak_dxk_1[i+1])
		
		for_prev = np.multiply(from_next,dak_dpk)
		for_prev = np.tensordot(for_prev,np.transpose(wu),1)

		state_inps = np.array(self.state_inp)
		gradient = np.multiply(gradient,self.activation.derivate(potentials))
		gradient = np.multiply(np.transpose(state_inps,[0,1,3,2]), gradient)

		return np.sum(gradient,(0,1)), for_prev

	def optimize_weights(self,delta):
		self.weights = self.weights - delta
	def reset(self):
		self.state_inp = []
		self.potentials = []
		self.current_state = np.zeros((self.batch_size, 1, self.size))
		self.weights = np.random.randn(self.size+self.input_size, self.size)
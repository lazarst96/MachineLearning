from Layer import *



class DenseLayer(Layer):
	def __init__(self,input_size,size,bias,activationfunction : ActivationFunction):
		self.size = size
		self.input_size = input_size
		self.bias = bias
		if bias:
			self.input_size = self.input_size + 1
		self.activation = activationfunction
		self.weights = np.random.randn(self.input_size, self.size)
		self.input_batches = []
		self.potentials = []

	def forward(self,input_batch):
		if self.bias:
			input_batch=np.dstack((input_batch,np.ones( (input_batch.shape[0],input_batch.shape[1],1) )))

		self.input_batches.append(input_batch)
		potential = np.tensordot(input_batch,self.weights,1)
		self.potentials.append(potential)
		return self.activation.function(potential)

	#sledeci gradient misli se u forward smeru
	def backward(self,from_next):
		potentials = np.array(self.potentials)
		gradient = np.multiply(from_next,self.activation.derivate(potentials))

		if self.bias:
			weights=self.weights[0:-1,:]
		else:
			weights = self.weights

		for_prev = np.tensordot(gradient,np.transpose(weights),1)

		input_batches = np.array(self.input_batches)
		
		gradient = np.multiply(np.transpose(input_batches,[0,1,3,2]), gradient)
		return np.sum(gradient,(0,1)), for_prev

	

	def backward_for_cross_entropy(self, from_next):
		gradient = from_next
		if self.bias:
			weights=self.weights[0:-1,:]
		else:
			weights = self.weights

		for_prev = np.tensordot(gradient,np.transpose(weights),1)
		input_batches = np.array(self.input_batches)
		gradient = np.multiply(np.transpose(input_batches,[0,1,3,2]), gradient)
		return np.sum(gradient,(0,1)), for_prev

	def reset_state(self):
		self.input_batches = []
		self.potentials = []

	def reset(self):
		self.weights = np.random.randn(self.input_size, self.size)
		self.input_batches = []
		self.potentials = []

	def optimize_weights(self,delta):
		self.weights = self.weights - delta
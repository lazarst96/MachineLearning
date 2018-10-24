from RecurrentLayer import *

class GRULayer(RecurrentLayer):
	def __init__(self,input_size,size, batch_size, activationfunction  : ActivationFunction = Tanh()):
		self.size = size
		self.input_size = input_size
		self.batch_size = batch_size
		self.input_size = self.input_size + 1

		self.activation = activationfunction
		self.sigmoid = Sigmoid()

		#tezine za reset gate
		self.wr = np.random.randn(self.size+self.input_size, self.size)
		#tezine za update gate
		self.wu = np.random.randn(self.size+self.input_size, self.size)
		#tezine za state candidate
		self.w = np.random.randn(self.size+self.input_size, self.size)

		self.current_state = np.zeros((self.batch_size, 1, self.size))
		self.state_inp = []
		self.states = [self.current_state]
		self.state_inp_fc = []
		self.reset_gate_potentials = []
		self.update_gate_potentials = []
		self.state_candidate_potentials = []
		self.states_candidate = []
		self.reset_gates= []
		self.update_gates= []
		
	def forward(self, input_batch):

		input_batch=np.dstack((input_batch,np.ones( (input_batch.shape[0],input_batch.shape[1],1) )))
		state_inp = np.dstack((input_batch, self.current_state))
		self.state_inp.append(state_inp)

		reset_gate = np.tensordot(state_inp,self.wr,1)
		update_gate = np.tensordot(state_inp,self.wu,1)
		self.reset_gate_potentials.append(reset_gate)
		self.update_gate_potentials.append(update_gate)


		reset_gate = self.sigmoid.function(reset_gate)
		update_gate = self.sigmoid.function(update_gate)
		self.reset_gates.append(reset_gate)
		self.update_gates.append(update_gate)

		state_inp_fc = np.dstack((input_batch, np.multiply(self.current_state, reset_gate)))
		self.state_inp_fc.append(state_inp_fc)


		state_candidate = np.tensordot(state_inp_fc,self.w,1)
		self.state_candidate_potentials.append(state_candidate)

		state_candidate = self.activation.function(state_candidate)
		self.states_candidate.append(state_candidate)

		self.current_state = np.multiply(self.current_state, update_gate) + np.multiply(state_candidate, (1-update_gate))
		self.states.append(self.current_state)
		return self.current_state

	def backward(self,from_next):
		# x je vector ulaza, h je stanje sloja, c je stanje kandidat
		state_inp = np.array(self.state_inp)
		h = np.array(self.states)
		state_inp_fc = np.array(self.state_inp_fc)
		r_potentials = np.array(self.reset_gate_potentials)
		u_potentials = np.array(self.update_gate_potentials)
		c_potentials = np.array(self.state_candidate_potentials)
		c = np.array(self.states_candidate)
		r = np.array(self.reset_gates)
		u = np.array(self.update_gates)


		wr_h=self.wr[self.input_size:,:]
		wu_h=self.wu[self.input_size:,:]
		w_h=self.w[self.input_size:,:]

		wr_x=self.wr[0:self.input_size-1,:]
		wu_x=self.wu[0:self.input_size-1,:]
		w_x=self.w[0:self.input_size-1,:]


		#izvod update gate-a po prethodnom stanju
		du_dh = np.tensordot(self.sigmoid.derivate(u_potentials), np.transpose(wu_h),1)
		#izvod reset gate-a po prethodnom stanju
		dr_dh = np.tensordot(self.sigmoid.derivate(r_potentials), np.transpose(wr_h),1)
		#izvod state candidate-a po prethodnom stanju
		dc_dh = np.tensordot(self.activation.derivate(c_potentials), np.transpose(w_h),1)*(r + h[0:-1]*dr_dh)
		#izvod stanja po prethodnom stanju
		dhi_dhi_1 = du_dh*h[0:-1] + u + (-du_dh)*c + (1-u)*dc_dh

		#uredjeni parcijalni izvod (suma gresaka od k do broj epoha) po stanju k
		gradient = from_next

		for i in range(from_next.shape[0]-2,-1,-1):
			gradient[i] += np.multiply(gradient[i+1], dhi_dhi_1[i+1])

		
		#izvod h(i-1)*c(i) po x(i)
		a = np.tensordot(from_next*h[0:-1]*self.sigmoid.derivate(u_potentials),np.transpose(wu_x) ,1)
		#izvod 
		b = - np.tensordot(from_next*c*self.sigmoid.derivate(u_potentials),np.transpose(wu_x) ,1)
		p = np.tensordot(np.tensordot(from_next*(1-u)*self.activation.derivate(c_potentials)*h[0:-1],np.transpose(w_h),1)*self.sigmoid.derivate(r_potentials), np.transpose(wr_x),1)
		print("p shape",p.shape)
		q = np.tensordot(from_next*(1-u)*self.activation.derivate(c_potentials), np.transpose(w_x),1)
		for_prev = a +b  + q + p





		return , for_prev





		
	def optimize_weights(self,delta):
		self.wr -= delta[0:input_size+size]
		self.wu -= delta[input_size+size:2*(input_size+size)]
		self.w -= delta[2*(input_size+size):3*(input_size+size)]
	def reset(self):
		self.wr = np.random.randn(self.size+self.input_size, self.size)
		self.wu = np.random.randn(self.size+self.input_size, self.size)
		self.w = np.random.randn(self.size+self.input_size, self.size)

		self.current_state = np.zeros((self.batch_size, 1, self.size))
		self.state_inp = []
		self.states = [self.current_state]
		self.state_inp_fc = []
		self.reset_gate_potentials = []
		self.update_gate_potentials = []
		self.state_candidate_potentials = []
		self.states_candidate = []
		self.reset_gates= []
		self.update_gates= []

	def reset_state(self):
		self.current_state = np.zeros((self.batch_size, 1, self.size))
		self.state_inp = []
		self.states = [self.current_state]
		self.state_inp_fc = []
		self.reset_gate_potentials = []
		self.update_gate_potentials = []
		self.state_candidate_potentials = []
		self.states_candidate = []
		self.reset_gates= []
		self.update_gates= []


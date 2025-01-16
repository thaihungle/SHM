import torch
from torch import nn
import random

class SHM(nn.Module):
	def __init__(
		self, 
		input_size,
		output_size,
		mem_size,#H in the paper
		add_space = 7
	):
		super().__init__()
			
		self.add_space = add_space
		self.L =  2**self.add_space
		self.mem_size = mem_size 
		self.key = nn.Linear(input_size, self.mem_size, bias=False)
		self.query = nn.Linear(input_size, self.mem_size, bias=False)
		self.value = nn.Linear(input_size, self.mem_size, bias=False)
		self.vc_net = nn.Linear(input_size, self.mem_size, bias=False)
		self.eta = nn.Linear(input_size, 1, bias=False)
		self.theta_matrix = nn.Parameter(torch.rand(self.L, self.mem_size))
		torch.nn.init.xavier_uniform(self.theta_matrix)
		self.out = nn.Linear(mem_size, output_size)
		self.norm = nn.LayerNorm(input_size)
	

	def initial_state(self, x):
		return torch.zeros(x.shape[0], self.mem_size, self.mem_size).to(x.device)


	def retrieve_theta(self, x, theta_matrix):
		B, T, D = x.shape
		a = torch.empty(x.shape[0], x.shape[1]).uniform_(0, 1).long().to(x.device)  # generate a uniform random matrix with range [0, 1]
		attention = torch.nn.functional.one_hot(a, num_classes=self.L).float()
		theta = torch.einsum("ij, bti -> btj", theta_matrix, attention)
		return theta


	def matrix_memory_update(self, x, state):
		x = self.norm(x)
		B, T, d = x.shape #B, T, d
		
		K = self.key(x).relu()
		Q = self.query(x).relu()
		
		
		eta1 = torch.sigmoid(self.eta(x))
		vc = self.vc_net(x)
		theta = self.retrieve_theta(x, self.theta_matrix)
		C = torch.einsum("bti, btj -> btij",  theta, vc)
		C = 1.0+torch.tanh(C)

		K = K / (1e-5 + K.sum(dim=-1, keepdim=True))
		Q = Q / (1e-5 + Q.sum(dim=-1, keepdim=True))
		V = self.value(x)

		state = state.squeeze(1)
		states = []

		#Memory writting: naive sequential implementation
		for t in range(T):
			state = state*C[:,t] + torch.einsum("bi, bj -> bij", eta1[:,t]*V[:,t],  K[:,t,:])
			states.append(state)
		state = torch.stack(states, dim=1)

		#Simple Memory reading: 
		y = torch.einsum("btij, btj -> bti", state, Q)
		return y, state

	def forward(self, x):
		B, T, d = x.shape #B, T, d
		M0 = self.initial_state(x)
		z, M = self.matrix_memory_update(x, M0)
		y = self.out(z)
		return y


if __name__ == '__main__':
	batch, length, dim = 2, 64, 16
	x = torch.randn(batch, length, dim).to("cuda")
	model = SHM(
		# This module uses roughly 3 * expand * d_model^2 parameters
		input_size=dim, # Model dimension d_model
		mem_size=16,  # SSM state expansion factor
	   output_size=32
	).to("cuda")
	y = model(x)

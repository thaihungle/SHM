import torch
from torch import nn
import random
from typing import Optional, Tuple
import math


class SHMAgent(nn.Module):
	
	def __init__(
		self,
		input_size,
		hidden_size,
		num_layers=1,
		batch_first=False,
		bias=True,
	):
		super().__init__()
		self.add_space = 7
		self.L =  2**self.add_space
		self.mem_size = 24
		self.skip_interval = 1
		self.key = nn.Linear(input_size, self.mem_size, bias=False)
		self.query = nn.Linear(input_size, self.mem_size, bias=False)
		self.value = nn.Linear(input_size, self.mem_size, bias=False)
		self.vc_net = nn.Linear(input_size, self.mem_size, bias=False)
		self.eta = nn.Linear(input_size, 1, bias=False)

		self.shortcut = nn.Linear(input_size, self.mem_size)
		self.out = nn.Linear(self.mem_size, hidden_size)
		self.theta_matrix = nn.Parameter(torch.rand(self.L, self.mem_size))
		torch.nn.init.xavier_uniform(self.theta_matrix)
		self.norm = nn.LayerNorm(input_size)


	def get_initial_state(self):
		return torch.zeros(1, 1, self.mem_size, self.mem_size).to("cuda")

	def retrieve_theta(self, x, theta_matrix):
		B, T, D = x.shape
		ri = torch.randint(0, self.L, (B*T,)).to(x.device) 
		rows = torch.index_select(theta_matrix, 0, ri)
		theta =rows.view(B, T, -1)
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


		kv = torch.einsum("bti, btj -> btij", eta1*V,  K)

		state = state.squeeze(1)
		states = []
	
		for t in range(T):
			if t%self.skip_interval==0:
				state = state*C[:,t]
			state = state + kv[:,t]
			state = state.clamp(-100,100)
			states.append(state)
		state = torch.stack(states, dim=1)


		y = torch.einsum("btij, btj -> bti", state, Q)
		y = y + self.shortcut(x) 
		return self.out(y), state[:, -1].unsqueeze(1)

	def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
		T, B, d = x.shape
		if state is not None:
			z, state = self.matrix_memory_update(x.transpose(0, 1),  state)
		else:
			state = torch.zeros(B, 1, self.mem_size, self.mem_size).to(x.device)
			z, state = self.matrix_memory_update(x.transpose(0, 1), state)
		return z.transpose(0, 1), state




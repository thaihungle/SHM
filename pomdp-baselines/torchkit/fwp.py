import torch
from torch import nn
import random
from typing import Optional, Tuple

class FWPBlock(nn.Module):
	"""
	The building block in the fast weight transformers paper. This is
	a form of linear transformer.

	Note that this is a "simplified" version of FWP without the delta
	update rule and DPFP. We use ReLU instead of DPFP.

	Inputs:
		input_size: Size of input feature dim
		hidden_size: Size of key/query/value space
		aggregator: Which type of aggregation to use
		sum_normalization: Whether to use the sum normalization described
			in the paper
		feed_forward: Whether to apply a perceptron to the output
		residual: Whether to apply a residual connection from input to output
	"""

	def __init__(
		self,
		input_size,
		hidden_size,
		sum_normalization=True,
		feed_forward=True,
		residual=True,
	):
		super().__init__()
		self.key = nn.Linear(input_size, hidden_size, bias=False)
		self.query = nn.Linear(input_size, hidden_size, bias=False)
		self.value = nn.Linear(input_size, hidden_size, bias=False)
		self.beta = nn.Linear(input_size, 1, bias=False)
		self.norm = nn.LayerNorm(input_size)
		self.sum_normalization = sum_normalization
		self.feed_forward = feed_forward
		self.residual = residual

		if self.residual:
			self.shortcut = nn.Linear(input_size, hidden_size)

	def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
		"""
		Inputs:
			x: [B, T, F]
			state: [B, 1, F]
		Outputs:
			y: [B, T, D]
			state: [B, T, F]
		"""
		x = self.norm(x)
		shape = x.shape #B, T, d
		K = self.key(x).relu()
		Q = self.query(x).relu()
		V = self.value(x)
		beta = torch.sigmoid(self.beta(x))
		if self.sum_normalization:
			K = K / (1e-5 + K.sum(dim=-1, keepdim=True))
			Q = Q / (1e-5 + Q.sum(dim=-1, keepdim=True))

		# kv = torch.einsum("bti, btj -> btij", V, K)
		T = shape[1]
		states = []
		state = state.squeeze(1)
		for t in range(T):
			Vbar = torch.einsum("bij, bj -> bi", state, K[:,t,:])
			state = state + torch.einsum("bi, bj -> bij", beta[:,t]*(V[:,t,:]-Vbar),  K[:,t,:])
			states.append(state)
		states = torch.stack(states, dim=1)


		y = torch.einsum("btij, btj -> bti", states, Q)


		if self.residual:
			y = y + self.shortcut(x)
		return y, states


class FWPAgent(nn.Module):
	
	def __init__(
		self,
		input_size,
		hidden_size,
		num_layers=1,
		batch_first=False,
		bias=True
	):
		super().__init__()
		self.mem_size=64
		self.core = FWPBlock(
			input_size=input_size,
			hidden_size=self.mem_size,
			sum_normalization=True,
		)
		self.unmap = nn.Linear(self.mem_size, hidden_size)

	def get_initial_state(self):
		return torch.zeros(1, 1, self.mem_size, self.mem_size).to("cuda")

	def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
		T, B, d = x.shape
		if state is not None:
			z, state = self.core(x.transpose(0, 1), state)
		else:
			state = torch.zeros(B, 1, self.mem_size, self.mem_size).to(x.device)
			z, state = self.core(x.transpose(0, 1), state)
		z = self.unmap(z)
		return z.transpose(0, 1), state


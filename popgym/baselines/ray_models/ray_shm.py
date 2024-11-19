from typing import List, Tuple
import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn
from popgym.baselines.ray_models.base_model import BaseModel
import random

class SHMAgent(BaseModel):
	

	MODEL_CONFIG = {
		# Whether to use the sum normalization over the key/query term
		# as in the paper
		"sum_normalization": True,
		# Which positional embedding to use
		"embedding": "sine",
		# "embedding": None,
		# Which cumulative aggregator to use. Only sum is used in the paper.
		# This can be sum or max
		"aggregator": "sum",
		"mem_size": 16,
		"post_size": 1024,
	}

	def __init__(
		self,
		obs_space: gym.spaces.Space,
		action_space: gym.spaces.Space,
		num_outputs: int,
		model_config: ModelConfigDict,
		name: str,
		**custom_model_kwargs,
	):
		super().__init__(obs_space, action_space, num_outputs, model_config, name)
		
		
		self.add_space = 7
		self.L =  2**self.add_space
		self.mem_size = self.cfg["mem_size"] #H in the paper
		self.key = nn.Linear(self.cfg["preprocessor_output_size"], self.mem_size, bias=False)
		self.query = nn.Linear(self.cfg["preprocessor_output_size"], self.mem_size, bias=False)
		self.value = nn.Linear(self.cfg["preprocessor_output_size"], self.mem_size, bias=False)
		self.vc_net = nn.Linear(self.cfg["preprocessor_output_size"], self.mem_size, bias=False)
		self.eta = nn.Linear(self.cfg["preprocessor_output_size"], 1, bias=False)
		self.theta_matrix = nn.Parameter(torch.rand(self.L, self.mem_size))
		torch.nn.init.xavier_uniform(self.theta_matrix)
		self.shortcut = nn.Linear(self.cfg["preprocessor_output_size"], self.mem_size)
		self.out = nn.Linear(self.cfg["mem_size"], self.cfg["post_size"])
		self.norm = nn.LayerNorm(self.cfg["preprocessor_output_size"])
	

	def initial_state(self) -> List[TensorType]:
		return [torch.zeros(1, self.mem_size, self.mem_size)]


	def retrieve_theta(self, x, theta_matrix):
		B, T, D = x.shape
		a = torch.empty(x.shape[0], x.shape[1]).uniform_(0, 1).long().to(x.device)  # generate a uniform random matrix with range [0, 1]
		attention = torch.nn.functional.one_hot(a, num_classes=self.L).float()
		theta = torch.einsum("ij, bti -> btj", theta_matrix, attention)
		return theta


	def matrix_memory_update(self, x, z, state):
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

		#Memory reading: following original FWP code of the benchmark
		y = torch.einsum("btij, btj -> bti", state, Q)
		y = y + self.shortcut(x) 
		return y, state

	def forward_memory(
		self,
		z: TensorType,
		state: List[TensorType],
		t_starts: TensorType,
		seq_lens: TensorType,
	) -> Tuple[TensorType, List[TensorType]]:
		M = state[0]
		x = z
		y, M = self.matrix_memory_update(x, z, M)
		state = [M[:, -1].reshape(z.shape[0], 1, self.mem_size, self.mem_size)]  # type: ignore
		z = self.out(y)
		return z, state

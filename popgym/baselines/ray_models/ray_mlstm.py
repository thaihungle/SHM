from typing import List, Tuple

import torch
from torch import nn

import math
import json
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

	
	
		
class mLSTM(nn.Module):
	def __init__(self, input_size, mem_dim):
		super(mLSTM, self).__init__()
		self.input_size = input_size
		self.mem_dim = mem_dim
		self.Wq = nn.Parameter(torch.randn(input_size, mem_dim))
		self.bq = nn.Parameter(torch.randn(1, mem_dim))
		self.Wk = nn.Parameter(torch.randn(input_size, mem_dim))
		self.bk = nn.Parameter(torch.randn(1, mem_dim))
		self.Wv = nn.Parameter(torch.randn(input_size, mem_dim))
		self.bv = nn.Parameter(torch.randn(1, mem_dim))
		self.wi = nn.Parameter(torch.randn(input_size, 1))
		self.bi = nn.Parameter(torch.randn(1))
		self.wf = nn.Parameter(torch.randn(input_size, 1))
		self.bf = nn.Parameter(torch.randn(1))
		self.Wo = nn.Parameter(torch.randn(input_size, mem_dim))
		self.bo = nn.Parameter(torch.randn(1, mem_dim))
		self.reset_parameters()

	def reset_parameters(self):
		for p in self.parameters():
			if p.data.ndimension() >= 2:
				nn.init.xavier_uniform_(p.data)
			else:
				nn.init.zeros_(p.data)

	def forward(self, x, states):
		(C_prev, n_prev) = states
		T, B, d = x.shape
		qt = torch.matmul(x,self.Wq) + self.bq
		kt = (1 / math.sqrt(self.mem_dim)) * (torch.matmul(x, self.Wk) + self.bk)
		vt = torch.matmul(x, self.Wv) + self.bv

		it = torch.exp(torch.matmul(x, self.wi) + self.bi).unsqueeze(-1)
		ft = torch.sigmoid(torch.matmul(x, self.wf) + self.bf).unsqueeze(-1)

		vt = vt.squeeze()
		kt = kt.squeeze()

		C = ft * C_prev + it * torch.matmul(vt.unsqueeze(-1), kt.unsqueeze(-2))
		n = ft.squeeze(-1) * n_prev + it.squeeze(-1) * kt
		max_nqt = torch.max(torch.abs(torch.matmul(n.unsqueeze(-2), qt.unsqueeze(-1))), torch.ones(T, B, 1, 1).to(x.device))
		h_tilde = torch.matmul(C, qt.unsqueeze(-1)).squeeze(-1) / max_nqt.squeeze(-1)
		ot = torch.sigmoid(torch.matmul(x, self.Wo) + self.bo)
		ht = ot * h_tilde

		return ht, (C, n)

	def init_hidden(self, T, B):
		return (torch.zeros(T, B, self.mem_dim, self.mem_dim).to("cuda"),
				torch.zeros(T, B, self.mem_dim).to("cuda"))



class xLSTMAgent(nn.Module):
   


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
		self.mem_size = self.cfg["mem_size"]
		self.input_size = self.cfg["preprocessor_output_size"]
		self.out = nn.Linear(self.mem_size, self.cfg["post_size"])
		self.core = mLSTM(input_size, self.mem_size)
	

    def initial_state(self) -> List[TensorType]:
        return [torch.zeros(1, self.cfg["mem_size"], self.cfg["mem_size"]), torch.zeros(1, self.cfg["mem_size"])]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        memory = (state[0],state[1])
        z, memory = self.core(z, memory)
        z = self.out(z)
        # State expected to be list
        state = [memory[0],memory[1]]  # type: ignore

        return z, state

	


   

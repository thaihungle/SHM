from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn
from popgym.baselines.models.mamba import MambaBlock
from popgym.baselines.ray_models.base_model import BaseModel


class RayMamba(BaseModel):
   
    MODEL_CONFIG = {
        # Number of recurrent hidden layers in encoder/decoder
        "num_recurrent_layers": 1,
        "benchmark": False,
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
        # Need to define self.core
        self.out = nn.Linear(self.cfg["preprocessor_output_size"], self.cfg["hidden_size"])
 
        
        self.core = MambaBlock(self.cfg["preprocessor_output_size"], self.cfg["hidden_size"])
               

    def initial_state(self) -> List[TensorType]:
        return [torch.zeros(1, self.cfg["preprocessor_output_size"]*2, self.cfg["hidden_size"])]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        memory = state[0]
        z, memory = self.core(z, memory)
        z = self.out(z)
        # State expected to be list
        state = [memory]  # type: ignore

        return z, state

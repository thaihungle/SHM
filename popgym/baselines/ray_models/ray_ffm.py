from typing import List, Tuple

import gymnasium as gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.ray_models.base_model import BaseModel
from popgym.baselines.models.ffm import FFM


class RayFFM(BaseModel):
    r"""The gated recurrent unit from

    .. code-block:: text

        @article{chung_empirical_2014,
            title = {
                Empirical evaluation of gated recurrent neural
                networks on sequence modeling
            },
            journal = {arXiv preprint arXiv:1412.3555},
            author = {
                Chung, Junyoung and Gulcehre, Caglar and Cho,
                KyungHyun and Bengio, Yoshua
            },
            year = {2014},
        }
    """

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
        self.core = FFM(input_size=self.cfg["preprocessor_output_size"], 
        hidden_size=self.cfg["hidden_size"], 
        memory_size=self.cfg["mem_size"], 
        context_size=4, 
        output_size=self.cfg["post_size"])

    def initial_state(self) -> List[TensorType]:
        return [self.core.initial_state()]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        memory = state[0].to(torch.complex64).squeeze(1)
        z, memory = self.core(z, memory)

        
        state = [memory.unsqueeze(1)]  # type: ignore

        return z, state

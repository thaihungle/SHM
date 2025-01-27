import torch.nn as nn
from .ffm import FFMModel
from .fwp import FWPAgent
from .gpt2 import GPT2Agent
from .mamba import MambaAgent
from .xlstm import xLSTMAgent
from .shm import SHMAgent


LSTM_name = "lstm"
GRU_name = "gru"
FFM_name = "ffm"
FWP_name = "fwp"
GPT_name = "gpt"
S6_name = "s6"
xLSTM_name = "xlstm"
SHM_name = "shm"

RNNs = {
    LSTM_name: nn.LSTM,
    GRU_name: nn.GRU,
    FFM_name: FFMModel,
    FWP_name: FWPAgent,
    GPT_name: GPT2Agent,
    S6_name: MambaAgent,
    xLSTM_name: xLSTMAgent,
    SHM_name: SHMAgent
}

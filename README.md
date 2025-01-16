<div align="center">
  <img src="assets/shm_logo.png" height=200>
  <h1><b> Stable Hadamard Memory: A Unified Linear Memory Framework </b></h1>
</div>

<div align="center">

[**English**](./README.md) 

</div>

<div align="center">

[![LICENSE](https://img.shields.io/badge/License-Apache-green)](https://github.com/thaihungle/SHM/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange)](https://pytorch.org/)
[![Popgym](https://img.shields.io/badge/Power_by-Popgym-blue)](https://github.com/proroklab/popgym)
[![Pomdp-Baselines](https://img.shields.io/badge/Power_by-Pomdp_Baselines-pink)](https://github.com/twni2016/pomdp-baselines)


</div>

<div align="center">

🚀 [**Getting Started**](#install) **|**
🔧 [**Usage**](#usage) **|**
🎯 [**Benchmarks**](#bench) **|**
🧠 [**Baselines**](#baselines) **|**
🤝 [**Todo**](#todo)
</div>

**Stable Hadamard Memory (SHM)** framework delivers a breakthrough in scalable and robust memory for deep learning models. Using the Hadamard product for updates and calibration, it ensures stable gradient flows while avoiding issues like vanishing or exploding gradients. 
🎉 SHM excels at long-term reasoning due to its attention-free, parallelizable design, and linear complexity, making it ideal for large-scale tasks.
✨ If you find SHM helpful, please share your feedback, cite our work, and give it a ⭐. Your support means a lot! 

**Why SHM?**
- SHM provides a stable and efficient approach to neural memory construction in deep sequence models, offering a strong foundation for advanced neural architectures.
- SHM is designed to be flexible and adaptable, making it easy to integrate into a wide range of applications and research workflows.
- SHM math is simple, yet generic:
<div align="center">
  <img src="https://github.com/user-attachments/assets/328189d0-e26f-40b0-9e48-980b0bb80f5e" height=100>
</div>

**Special cases of SHM:**
- [SSM](https://github.com/state-spaces/mamba): $M_t$, $C_t$, and $U_t$ are vectors
- [Linear Attention](https://github.com/lucidrains/linear-attention-transformer): $C_t=1$
- [mLSTM](https://github.com/NX-AI/xlstm): $C_t$ is a scalar


📜 For more details, check out our [paper](https://arxiv.org/abs/2410.10132) and [blogs](https://open.substack.com/pub/hungleai/p/stable-hadamard-memory-the-unified?r=3an4d1&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true).
Please feel free to let me know your suggestions. We're constantly working to improve and expand the framework.

> [!IMPORTANT]
> If you find this repository helpful for your work, please consider citing as follows:
>
> ```LaTeX
> @article{le2024stable,
>  title={Stable Hadamard Memory: Revitalizing Memory-Augmented Agents for Reinforcement Learning},
>  author={Le, Hung and Do, Kien and Nguyen, Dung and Gupta, Sunil and Venkatesh, Svetha},
>  journal={arXiv preprint arXiv:2410.10132},
>  year={2024}
> }
> ```
>

## <a name="install"></a> 🚀 Installation and Quick Start

#### ⏬ Cloning the Repository

First, clone the SHM repository:

```bash
cd /path/to/your/project
git clone https://github.com/thaihungle/SHM.git
```

#### 💿 Installing Dependencies

Python 3.8 or higher is recommended. If you use GPUs, CUDA 11 or higher is recommended. 
After ensuring the CUDA driver is installed correctly, you can install the other dependencies. 

We recommend setting up separate dependencies for each benchmark.

#### Example Setup for [POPGym](https://github.com/proroklab/popgym) benchmark: Python 3.8 + PyTorch 2.4.0

```bash
# Install Python
conda create -n SHM python=3.8
conda activate SHM
# Install other dependencies
pip install -r popgym_requirements.txt
```

#### Example Setup for [Pomdp-baselines](https://github.com/twni2016/pomdp-baselines) benchmark: Python 3.8 + PyTorch 2.4.0

```bash
# Install Python
conda create -n SHM python=3.8
conda activate SHM
# Install other dependencies
pip install -r pompd_requirements.txt
```
## <a name="usage"></a> 🔧 Usage
SHM can be used as an independent Pytorch module:
``` python
import torch
from shm import SHM

batch, length, dim = 2, 64, 16
# remove to("cuda") if you use CPU
x = torch.randn(batch, length, dim).to("cuda")
model = SHM(
  input_size=dim, 
  mem_size=16,  #H in the paper
  output_size=32
).to("cuda")
y = model(x)
```

Implementation details of the SHM module can be found in [shm.py](https://github.com/thaihungle/SHM/blob/main/shm.py). 
Just so you know, when we adapt to specific tasks, we can slightly modify the implementation to follow the common practice (e.g., add residual shortcut).

## <a name="bench"></a> 🎯 Benchmarks

#### ☝️ POPGym
[POPGym](https://github.com/proroklab/popgym) is designed to benchmark memory in deep reinforcement learning. 
Here, we focus on the most memory-intensive tasks:
- Autoencode
- Battleship
- Concentration
- RepeatPrevious
  
Each task consists of 3 modes of environments: easy, medium, and hard. 

**Example easy training using SHM with a memory size of 128:** 
```
python train_popgym.py --env AutoencodeEasy --model shm --m 128
python train_popgym.py --env BattleshipEasy --model shm --m 128
python train_popgym.py --env ConcentrationEasy --model shm --m 128
python train_popgym.py --env RepeatPreviousEasy --model shm --m 128
```
**Example hard training using SHM with a memory size of 32:** 
```
python train_popgym.py --env AutoencodeHard --model shm --m 32
python train_popgym.py --env BattleshipHard --model shm --m 32
python train_popgym.py --env ConcentrationHard --model shm --m 32
python train_popgym.py --env RepeatPreviousHard --model shm --m 32
```

**Results and Logs**

See folder ./results_popggym for Popgym's outputs and logs (we support Tensorboard!). You should be able to reproduce results like this:
<div align="center">
  <img src="https://github.com/user-attachments/assets/32d0b42c-4754-4776-be01-8965740962ad" height=300>
</div>

**Hyperparameters**

We follow the well-established hyperparameters set by POPGym. We only tune the memory-related hyperparameters:
- $m$: memory size for matrix-based memory models such as SHM
- $h$: hidden size for vector-based memory models such as GRU

For other hyperparameters, see [train_popgym.py](https://github.com/thaihungle/SHM/blob/main/train_popgym.py).

#### ✌️ Pomdp-baselines

TBU

## <a name="baselines"></a> 🧠 Baselines

#### ☝️ POPGym

In addition to default POPGym baselines. We have added the following models:
- [SHM](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_shm.py)
- [Mamba (S6)](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_mamba.py)
- [mLSTM](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_mlstm.py)  

To run experiments with baselines, please refer to  [train_popgym.py](https://github.com/thaihungle/SHM/blob/main/train_popgym.py) to add the baseline calls. 
Then, run the training command. 

**Example easy training using GRU with different hidden sizes:** 
```
python train_popgym.py --env AutoencodeEasy --model gru --h 256
python train_popgym.py --env AutoencodeEasy --model gru --h 512
python train_popgym.py --env AutoencodeEasy --model gru --h 1024
```

<details><summary>Other baselines</summary>
  
- [MLP](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_mlp.py)
- [MLP (frame stacked)](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_framestack.py)
- [RNN](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_elman.py)
- [GRU](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_gru.py)
- [LSTM](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_lstm.py)
- [CNN](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_frameconv.py)
- [iRNN](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_indrnn.py)
- [LMU](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_lmu.py)
- [DNC](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_doffnc.py)
- [FWP](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_fwp.py)
- [Linear Attention](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_linear_attention.py)
- [S4](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_s4d.py)
- [FFM](https://github.com/thaihungle/SHM/blob/main/popgym/baselines/ray_models/ray_ffm.py)

</details>

#### ✌️ Pomdp-baselines

TBU


## <a name="todo"></a> 🤝 Things to Do
- [X] POPgym Tasks
- [ ] Pomdp-baseline Tasks
- [ ] Time-series Tasks
- [ ] LLM Tasks

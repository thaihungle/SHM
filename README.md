<div align="center">
  <img src="assets/shm_logo.png" height=200>
  <h3><b> Stable Hadamard Memory: A Unified Linear Memory Framework </b></h3>
</div>

<div align="center">

[**English**](./README.md) 

</div>

<div align="center">

[![LICENSE](https://img.shields.io/badge/License-MIT-green)](https://github.com/thaihungle/SHM/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-orange)](https://pytorch.org/)
[![Popgym](https://img.shields.io/badge/Power_by-Popgym-blue)]([https://pytorch.org/](https://github.com/proroklab/popgym))


</div>

<div align="center">

🎉 [**Getting Started**](./tutorial/getting_started.md) **|**
📦 [**Tasks**](./tutorial/dataset_design.md) **|**
🧠 [**Model**](./tutorial/model_design.md) **|**
📜 [**Baselines**](./baselines/)

</div>

# SHM
Source code for Stable Hadamard Memory paper. 
Find the SHM implementation at popgym/baselines/ray_models/ray_shm.py  
The code is based on: https://github.com/proroklab/popgym 


### Setup
python 3.8  
```
pip install -r requirements.txt   
```

### Run Training
```
python train.py --env AutoencodeEasy --model shm --m 128 
```

See folder ./results for outputs and logs. 

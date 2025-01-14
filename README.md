<div align="center">
  <img src="assets/Basic-TS-logo-for-white.png#gh-light-mode-only" height=200>
  <img src="assets/Basic-TS-logo-for-black.png#gh-dark-mode-only" height=200>
  <h3><b> Stable Hadamard Memory: A Unified Linear Memory Framework </b></h3>
</div>

<div align="center">

[**English**](./README.md) 

</div>

<div align="center">

[![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch)
[![LICENSE](https://img.shields.io/github/license/zezhishao/BasicTS.svg)](https://github.com/zezhishao/BasicTS/blob/master/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-orange)](https://pytorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange)](https://pytorch.org/)
[![python lint](https://github.com/zezhishao/BasicTS/actions/workflows/pylint.yml/badge.svg)](https://github.com/zezhishao/BasicTS/blob/master/.github/workflows/pylint.yml)

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

<div align="center">
  <img src="assets/shm_logo.png" height=200>
  <h2><b> Stable Hadamard Memory: A Unified Linear Memory Framework </b></h2>
</div>

<div align="center">

[**English**](./README.md) 

</div>

<div align="center">

[![LICENSE](https://img.shields.io/badge/License-MIT-green)](https://github.com/thaihungle/SHM/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-orange)](https://pytorch.org/)
[![Popgym](https://img.shields.io/badge/Power_by-Popgym-blue)](https://github.com/proroklab/popgym)
[![Pomdp-Baselines](https://img.shields.io/badge/Power_by-Pomdp_Baselines-pink)](https://github.com/twni2016/pomdp-baselines)


</div>

<div align="center">

🎉 [**Getting Started**](./tutorial/getting_started.md) **|**
📦 [**Tasks**](./tutorial/dataset_design.md) **|**
🧠 [**Model**](./tutorial/model_design.md) **|**
📜 [**Baselines**](./baselines/)

</div>

**Stable Hadamard Memory (SHM)** framework delivers a breakthrough in scalable and robust memory for deep learning models. Using the Hadamard product for updates and calibration, it ensures stable gradient flows while avoiding issues like vanishing or exploding gradients.
SHM excels at long-term reasoning due to its attention-free, parallelizable design, and linear complexity, making it ideal for large-scale tasks.

✨ If you find SHM helpful, feel free to share your feedback, cite our work, and give it a ⭐. Your support means a lot! 

**Why SHM?**
- SHM provides a stable and efficient approach to memory management in deep sequence models, offering a strong foundation for advanced neural architectures.
- SHM is designed to be flexible and adaptable, making it easy to integrate into a wide range of applications and research workflows.

📜 For more details and tutorials, check out our documentation or reach out with your suggestions. We're constantly working to improve and expand the framework.

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

import popgym
import ray
from torch import nn
import argparse
import torch
import numpy as np
import random

def config_gru():
	from popgym.baselines.ray_models.ray_gru import GRU

	# See what GRU-specific hyperparameters we can set
	print(GRU.MODEL_CONFIG)
	# Show other settable model hyperparameters like
	# what the actor/critic branches look like,
	# what hidden size to use,
	# whether to add a positional embedding, etc.
	print(GRU.BASE_CONFIG)
	# How long the temporal window for backprop is
	# This doesn't need to be longer than 1024
	bptt_size = 1024
	config = {
	"model": {
		"max_seq_len": bptt_size,
		"custom_model": GRU,
		"custom_model_config": {
		# Override the hidden_size from BASE_CONFIG
		# The input and output sizes of the MLP feeding the memory model
		"preprocessor_input_size": 128,
		"preprocessor_output_size": 64,
		"preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
		# this is the size of the recurrent state in most cases
		"hidden_size": 256,
		# We should also change other parts of the architecture to use
		# this new hidden size
		# For the GRU, the output is of size hidden_size
		"postprocessor": nn.Sequential(nn.Linear(256, 64), nn.ReLU()),
		"postprocessor_output_size": 64,
		# Actor and critic networks
		"actor": nn.Linear(64, 64),
		"critic": nn.Linear(64, 64),
		# We can also override GRU-specific hyperparams
		"num_recurrent_layers": 1,
		},
	},
	"num_gpus": args.fgpu,
	"num_workers": 2,
	# Some other rllib defaults you might want to change
	# See https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
	# for a full list of rllib settings
	#
	# These should be a factor of bptt_size
	"sgd_minibatch_size": bptt_size * 8,
	# Should be a factor of sgd_minibatch_size
	"train_batch_size": bptt_size * 64,
	# The environment we are training on
	"env": f"popgym-{args.env}-v0",
	# You probably don't want to change these values
	"rollout_fragment_length": bptt_size,
	"framework": "torch",
	"horizon": bptt_size,
	"batch_mode": "complete_episodes",
	}

	return config

def config_ffm():
	from popgym.baselines.ray_models.ray_ffm import RayFFM

	# See what GRU-specific hyperparameters we can set
	print(RayFFM.MODEL_CONFIG)
	# Show other settable model hyperparameters like
	# what the actor/critic branches look like,
	# what hidden size to use,
	# whether to add a positional embedding, etc.
	print(RayFFM.BASE_CONFIG)
	# How long the temporal window for backprop is
	# This doesn't need to be longer than 1024
	bptt_size = 1024
	config = {
	"model": {
		"max_seq_len": bptt_size,
		"custom_model": RayFFM,
		"custom_model_config": {
		# Override the hidden_size from BASE_CONFIG
		# The input and output sizes of the MLP feeding the memory model
		"preprocessor_input_size": 128,
		"preprocessor_output_size": 64,
		"preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
		# this is the size of the recurrent state in most cases
		"hidden_size": 64,
		# We should also change other parts of the architecture to use
		# this new hidden size
		# For the GRU, the output is of size hidden_size
		"mem_size": args.m,
 		"post_size": args.post_size,
		"postprocessor": nn.Sequential(nn.Linear(args.post_size, 64), nn.ReLU()),       
		"postprocessor_output_size": 64,
		# Actor and critic networks
		"actor": nn.Linear(64, 64),
		"critic": nn.Linear(64, 64),
		# We can also override GRU-specific hyperparams
		"num_recurrent_layers": 1,
		},
	},
	"num_gpus": args.fgpu,
	"num_workers": 2,
	# Some other rllib defaults you might want to change
	# See https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
	# for a full list of rllib settings
	#
	# These should be a factor of bptt_size
	"sgd_minibatch_size": bptt_size * 8,
	# Should be a factor of sgd_minibatch_size
	"train_batch_size": bptt_size * 64,
	# The environment we are training on
	"env": f"popgym-{args.env}-v0",
	# You probably don't want to change these values
	"rollout_fragment_length": bptt_size,
	"framework": "torch",
	"horizon": bptt_size,
	"batch_mode": "complete_episodes",
	}

	return config
	
def config_shm():
   
	from popgym.baselines.ray_models.ray_shm import SHMAgent

	# See what specific hyperparameters we can set
	print(SHMAgent.MODEL_CONFIG)
	# Show other settable model hyperparameters like
	# what the actor/critic branches look like,
	# what hidden size to use,
	# whether to add a positional embedding, etc.
	print(SHMAgent.BASE_CONFIG)
	# How long the temporal window for backprop is
	# This doesn't need to be longer than 1024
	
	bptt_size = 1024
	config = {
	"model": {
		"max_seq_len": bptt_size,
		"custom_model": SHMAgent,
		"custom_model_config": {
		# Override the hidden_size from BASE_CONFIG
		# The input and output sizes of the MLP feeding the memory model
		"preprocessor_input_size": 128,
		"preprocessor_output_size": 64,
		"preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
		"preprocessor2": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
		# this is the size of the recurrent state in most cases
		"hidden_size": args.h,
		# We should also change other parts of the architecture to use
		# this new hidden size
		"embedding_size": 128,
		"mem_size": args.m,
		"post_size": args.post_size,
		"postprocessor": nn.Sequential(nn.Linear(args.post_size, 64), nn.ReLU()),
		"postprocessor_output_size": 64,
		# Actor and critic networks
		"actor": nn.Linear(64, 64),
		"critic": nn.Linear(64, 64)
		},
	},
	"num_gpus": args.fgpu,
	"num_workers": 2,
	# Some other rllib defaults you might want to change
	# See https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
	# for a full list of rllib settings
	#
	# These should be a factor of bptt_size
	"sgd_minibatch_size": int(bptt_size*8*args.bscale),
	# Should be a factor of sgd_minibatch_size
	"train_batch_size": int(bptt_size*64*args.bscale),
	# The environment we are training on
	"env": f"popgym-{args.env}-v0",
	# You probably don't want to change these values
	"rollout_fragment_length": bptt_size,
	"framework": "torch",
	"horizon": bptt_size,
	"batch_mode": "complete_episodes",
	}

	return config
	

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Popgym Benchmark')

	# basic config
	parser.add_argument('--env', type=str,  default='AutoencodeEasy')
	parser.add_argument('--model', type=str,  default='gru')
	parser.add_argument('--nrun', type=int,  default=3)
	parser.add_argument('--bscale', type=float,  default=1.0)
	# parser.add_argument('--fgpu', type=float,  default=0.3)
	parser.add_argument('--h', type=int,  default=64)#hidden size
	parser.add_argument('--m', type=int,  default=72)#H in the paper
	parser.add_argument('--gpu', type=int,  default=1)
	parser.add_argument('--post_size', type=int,  default=1024)
	args = parser.parse_args()
	args.fgpu = 1.0/args.nrun

	print(args)

	if args.model=="gru":
		config = config_gru()
	elif args.model=="ffm":
		config = config_ffm()
	elif args.model=="shm":
		config = config_shm()
	ngpu = 1
	if args.gpu==0 or not torch.cuda.is_available():
		args.fgpu = 0
		ngpu = 0
		
	ray.init(ignore_reinit_error=True, num_cpus=10, num_gpus=ngpu)
	ray.tune.run("PPO", config=config, 
				num_samples = args.nrun, 
				stop={"timesteps_total": 15_000_000},
				local_dir=f"./results/{args.env}/{args.model}/")


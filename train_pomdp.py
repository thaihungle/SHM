import os 
import argparse
import sys
from os.path import dirname

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pomdp-Baselines Benchmark')
	# basic config
	parser.add_argument('--task', type=str,  default='meta', help='meta or credit')
	parser.add_argument('--env', type=str,  default='wind', help='wind or point_robot')
	parser.add_argument('--model', type=str,  default='shm', help='ffm, fwp, gpt, mlp, gru, s6, xlstm')
	parser.add_argument('--seed', type=int,  default=0)
	parser.add_argument('--nrun', type=int,  default=1)

	args = parser.parse_args()

	if args.model=="gru":
		args.model="rnn"
	for i in range(args.nrun):
		seed = args.seed + 10*i
		os.system(f'python pomdp-baselines/policies/main.py \
		--cfg pomdp-baselines/configs/{args.task}/{args.env}/{args.model}.yml --seed {seed}')
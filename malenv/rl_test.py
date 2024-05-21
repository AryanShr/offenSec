import math, random

import gym
import numpy as np
import sys
import os
np.random.seed(123)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.autograd as autograd 
from gym_malware.envs.malenv import MalwareEnv
from gym_malware.envs.utils import interface, pefeatures
from gym_malware.envs.controls import manipulate as manipulate
from collections import namedtuple, deque
from statistics import mean 
from gym_malware.envs.utils.interface import EnsembleBlackBoxDetector
from collections import namedtuple, deque
from statistics import mean 
import rl_train

env = MalwareEnv(random_sample=False,output_path="modified\\test")

from collections import deque

ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}
print(ACTION_LOOKUP)
# calculate epsilon


device = torch.device("cpu")

USE_CUDA = False

model = rl_train.DQN().to(device)
model.load_state_dict(torch.load('modified\\updatedEpsilon\\dqeaf\\rl-model-1600-2024-04-23.pt'))
model.eval()

# input_folder = "Virus test\Zeroaccess"
input_folder = "TestData\malware"
# output_folder = "test_modified\\dqeaf"
output_folder = "modified\\updatedEpsilon\\timepass"
onlyfiles = [f for f in os.listdir(input_folder)]

def test_model():

	
	T = 80 # total mutations allowed
	success = 0
	rn = rl_train.RangeNormalize(-0.5,0.5)
	fe = pefeatures.PEFeatureExtractor()
	episode = 0

	action_cnt = {'overlay_append': 100,
    'section_rename' : 0,
    'section_append' : 48,
    'imports_append' : 57,
    'upx_pack':0,
    'upx_unpack':19}
	for file in onlyfiles:
		try:
			with open(os.path.join(input_folder, file), 'rb') as infile:
				bytez = infile.read()
		except IOError:
			raise "Unable read the file"
		state = fe.extract( bytez )
		state_norm = rn(state)
		episode = episode + 1
		state_norm = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)
		prev = state_norm
		prev_score = None
		k = 1
		for mutation in range(1, T):
			
			actions = model.forward(state_norm)
			print(actions)

			action = torch.argmax(actions).item()
			action = ACTION_LOOKUP[action]
			prev_bytez = bytez
			bytez = manipulate.modify_without_breaking( bytez, [action] )
			# print("----------------")
			# print(prev_bytez==bytez)
			# print("---------------")

			new_label = interface.get_score_local( bytez )

			# if prev_score is not None and prev_score == new_label:
			# 	if k>=len(ACTION_LOOKUP):
			# 		k = 1
			# 	flat_tensor = actions.flatten()
			# 	action = torch.topk(flat_tensor, k).indices[k-1].item()
			# 	action = ACTION_LOOKUP[action]
			# 	prev_bytez = bytez
			# 	bytez = manipulate.modify_without_breaking( bytez, [action])
			# 	new_label = interface.get_score_local( bytez )
			# 	k+=1
			# else:
			# 	k = 1
			action_cnt[action] +=1
			prev_score = new_label
			print('episode : ' + str(episode))
			print('mutation : ' + str(mutation))
			print('test action : ' + str(action))
			print('new label : ' + str(new_label))
			state = fe.extract(bytez)
			state_norm = rn(state)
			state_norm = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)
			# print("-----------------------")
			# print(state_norm)
			# print(torch.equal(state_norm,prev))
			# prev = state_norm
			if(new_label < 0.90):
				with open(os.path.join(output_folder, file), mode='wb') as file1:
					file1.write(bytes(bytez))
				break
	# After all files are processed, plot the bar graph
	total_count = sum(action_cnt.values())

	# Calculate percentages
	action_pct = {k: v / total_count * 100 for k, v in action_cnt.items()}
	plt.barh(list(action_pct.keys()), list(action_pct.values()))
	plt.ylabel('Actions')
	plt.xlabel('Percentage')
	plt.title('Action percentages in test_model')
	plt.show()
if __name__ == "__main__":
	test_model()
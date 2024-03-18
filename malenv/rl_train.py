import warnings
warnings.filterwarnings("ignore")

import glob

from pathlib import Path
from tqdm import tqdm
from datetime import date
import os
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from statistics import mean 

import math, random

import gym
import numpy as np
np.random.seed(123)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gym_malware.envs.malenv import MalwareEnv
from gym_malware.envs.utils import interface, pefeatures
from gym_malware.envs.controls import manipulate as manipulate
from collections import namedtuple, deque
from statistics import mean 
from gym_malware.envs.utils.interface import EnsembleBlackBoxDetector

env = MalwareEnv(random_sample=False,output_path="modified")

from collections import deque
np.random.seed(123)

ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}
device = torch.device("cpu")



# prioritized replay buffer
class NaivePrioritizedBuffer(object):
	def __init__(self, capacity, prob_alpha=0.6):
		self.prob_alpha = prob_alpha
		self.capacity   = capacity
		self.buffer     = []
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.pos        = 0
		self.priorities = np.zeros((capacity,), dtype=np.float32)
	
	def push(self, state, action, reward, next_state, done):
		
		max_prio = self.priorities.max() if self.buffer else 1.0
		
		if len(self.buffer) < self.capacity:
			e = self.experience(state, action, reward, next_state, done)
			self.buffer.append(e)
		else:
			e = self.experience(state, action, reward, next_state, done)
			self.buffer[self.pos] = e
		
		self.priorities[self.pos] = max_prio
		self.pos = (self.pos + 1) % self.capacity
	
	def sample(self, batch_size, beta=0.4):
		if len(self.buffer) == self.capacity:
			prios = self.priorities
		else:
			prios = self.priorities[:self.pos]
		
		probs  = prios ** self.prob_alpha
		probs /= probs.sum()
		
		indices = np.random.choice(len(self.buffer), batch_size, p=probs)
		experiences = [self.buffer[idx] for idx in indices]
		
		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
		return (states, actions, rewards, next_states, dones, indices)
	   
	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in zip(batch_indices, batch_priorities):
			self.priorities[idx] = prio

	def __len__(self):
		return len(self.buffer)

# def update_epsilon(n):
# 	epsilon_start = 1.0
# 	epsilon = epsilon_start
# 	epsilon_final = 0.4
# 	epsilon_decay = 1000 # N from the research paper (equation #6)

# 	epsilon = 1.0 - (n/epsilon_decay)

# 	if epsilon <= epsilon_final:
# 		epsilon = epsilon_final

# 	return epsilon
	
def update_epsilon(n):
    epsilon_start = 1.0
    epsilon_final = 0.4
    epsilon_decay = 1000  # N from the research paper (equation #6)

    # Inverse square root decay
    epsilon = epsilon_start / np.sqrt(1 + n / epsilon_decay)

    if epsilon <= epsilon_final:
        epsilon = epsilon_final

    return epsilon


# create a dqn class
class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(env.observation_space.shape[0], 256),
			nn.ReLU(),
			nn.Linear(256, 64),
			nn.ReLU(),
			nn.Linear(64, env.action_space.n)
		)

	def forward(self, x):
		return self.layers(x)


	def chooseAction(self, observation, epsilon):
		rand = np.random.random()
		if rand > epsilon:
			# observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
			actions = self.forward(observation)
			# actions = F.softmax(actions, dim=1)
			print("actions",actions)
			action = torch.argmax(actions).item()
			# print("Actions")
			# actions = self.forward(observation)
			# actions = F.softmax(actions, dim=1)
			# # print("actions", actions)

			# # Select top 3 actions
			# top3_probs, top3_actions = torch.topk(actions, 3)

			# # Scale up the top 3 actions
			# scale_factor = torch.tensor([50, 30, 20])  # define your scale_factor
			# scaled_actions = top3_probs * scale_factor.float()
			# # print(scaled_actions)

			# # Get probabilities
			# probabilities = F.softmax(scaled_actions, dim=0)

			# # Create bins
			# bins = torch.cumsum(probabilities, dim=0)

			# # Get another random variable
			# rand_var = np.random.random()

			# # Check in which bin it's falling
			# action_idx = (rand_var < bins).nonzero(as_tuple=True)[0][0]

			# # print(action_idx.item())
			# # print(top3_actions)
			# # print(top3_actions.flatten()[action_idx.item()])

			# # Choose that action and map it to the original action space
			# action = top3_actions.flatten()[action_idx.item()].item()
			# print(action)
		else:
			action = np.random.choice(env.action_space.n)

		return action

replay_buffer = NaivePrioritizedBuffer(500000)

print("[*] Initilializing Neural Network model ...")
current_model = DQN().to(device)
target_model  = DQN().to(device)

optimizer = optim.Adam(current_model.parameters())

gamma = 0.99 # discount factor as mentioned in the paper

def update_target(current_model, target_model):
	target_model.load_state_dict(current_model.state_dict())

# TD loss
# def compute_td_loss(batch_size):
# 	state, action, reward, next_state, done, indices = replay_buffer.sample(batch_size, 0.4) 


# 	Q_targets_next = target_model(next_state).detach().max(1)[0].unsqueeze(1)
# 	Q_targets = reward + (gamma * Q_targets_next * (1 - done))
# 	Q_expected = current_model(state).gather(1, action)

# 	loss  = (Q_expected - Q_targets.detach()).pow(2)
# 	prios = loss + 1e-5
# 	loss  = loss.mean()
		
# 	optimizer.zero_grad()
# 	loss.backward()
# 	replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
# 	optimizer.step()
	
# 	return loss

def compute_td_loss(batch_size):
    state, action, reward, next_state, done, indices = replay_buffer.sample(batch_size, 0.4) 

    # Use the current model to select the action
    best_actions = current_model(next_state).argmax(1).unsqueeze(1)
    
    # Use the target model to calculate the Q-value for the best actions
    Q_targets_next = target_model(next_state).gather(1, best_actions).detach()
    
    Q_targets = reward + (gamma * Q_targets_next * (1 - done))
    Q_expected = current_model(state).gather(1, action)

    loss  = (Q_expected - Q_targets.detach()).pow(2)
    prios = loss + 1e-5
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    
    return loss



# normaliza the features
class RangeNormalize(object):
	def __init__(self, 
				 min_val, 
				 max_val):
		"""
		Normalize a tensor between a min and max value
		Arguments
		---------
		min_val : float
			lower bound of normalized tensor
		max_val : float
			upper bound of normalized tensor
		"""
		self.min_val = min_val
		self.max_val = max_val

	def __call__(self, *inputs):
		outputs = []
		for idx, _input in enumerate(inputs):
			_min_val = _input.min()
			_max_val = _input.max()
			a = (self.max_val - self.min_val) / (_max_val - _min_val)
			b = self.max_val- a * _max_val
			_input = (_input * a ) + b
			outputs.append(_input)
		return outputs if idx > 1 else outputs[0]

def main():
	print("[*] Starting training ...")
	D = 6000 #rl episodes
	T = 80 #mutations
	B = 1000 # as mentioned in the paper (number of steps before learning starts)
	batch_size = 32 # as mentioned in the paper (batch_size)
	losses = []
	rewards = [0]
	reward_ben = 20
	n = 0 #current training step
	rn = RangeNormalize(-0.5,0.5)
	check = False

	# malware_path =  "Data\malware\*"
	# file_paths = glob.glob(malware_path)
	# chosen_files= []
	# def get_random_file():
	# 	global file_paths, chosen_files
	# 	# Choose a random file
	# 	file_path = random.choice(file_paths)
	# 	# Check if the file has already been chosen
	# 	while file_path in chosen_files:
	# 		# If it has, choose another file
	# 		file_path = random.choice(file_paths)
	# 	# Add the chosen file to the list of chosen files
	# 	chosen_files.append(file_path)
	# 	return file_path

	

	fig, axs = plt.subplots(1,figsize=(10,8))  # Create a figure and a set of subplots
	for episode in range(1, D):
		malware_path =  "Data\malware\*"
		file_path = glob.glob(malware_path)[episode]
		print("Training: ", file_path)
		state = env.reset(file_path)
		state_norm = rn(state)
		state_norm = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)
		mutReward = 0
		for mutation in range(1, T):
			n = n + 1
			epsilon = update_epsilon(n)
			action = current_model.chooseAction(state_norm, epsilon)
			next_state, reward, done, _ = env.step(action)
			print("\t[+] Episode : " + str(episode) + " , Mutation # : " + str(mutation) + " , Mutation : " + str(ACTION_LOOKUP[action]) + " , Reward : " + str(reward))
			next_state_norm = rn(next_state) 
			next_state_norm = torch.from_numpy(next_state_norm).float().unsqueeze(0).to(device)

			if reward == 10.0:
				power = -((mutation-1)/T)
				reward = (math.pow(reward_ben, power))*100

			replay_buffer.push(state_norm, action, reward, next_state_norm, done)
			# mutReward += reward

			if len(replay_buffer) > B:
				loss = compute_td_loss(batch_size)
				print("Loss: ", loss.item())
				losses.append(loss.item())

			if done:
				break

			state_norm = next_state_norm

		print('\t[+] Episode Over')
		if n % 10 == 0:
			update_target(current_model, target_model)

		rl_save_model_interval = 10
		rl_output_directory = "modified/updatedEpsilon"
		if episode % rl_save_model_interval == 0:
			if not os.path.exists(rl_output_directory):
				os.mkdir(rl_output_directory)
				print("[*] model directory has been created at : " + str(rl_output_directory))
			torch.save(current_model.state_dict(), os.path.join(rl_output_directory, "rl-model-" + str(episode) + "-" +str(date.today()) + ".pt" ))
			print("[*] Saving model in models/ directory ...")
		axs.clear()
		axs.plot(losses, color='b')  # Plot the losses on the second graph
		axs.set_ylabel('Losses', color='b')  # Label the y-axis of the second graph
		# rewards.append(rewards[-1]+mutReward)
		# axs[1].clear()
		# axs[1].plot(rewards, color='g')  # Plot the rewards on the first graph
		# axs[1].set_ylabel('Rewards', color='g')  # Label the y-axis of the first graph
		plt.pause(0.01)
	torch.save(current_model.state_dict(), os.path.join(rl_output_directory, "rl-model-" + str(D) + "-" +str(date.today()) + ".pt" ))
	print("[*] Saving model in models/ directory ...")
	axs[0].plot(rewards, color='g')  # Plot the rewards on the first graph
	axs[1].plot(losses, color='b')  # Plot the losses on the second graph
	plt.show()
	
	
if __name__ == '__main__':
    main()
























import math, random
import glob
import gym
from pathlib import Path
from tqdm import tqdm
from datetime import date

import math, random
import numpy as np
np.random.seed(123)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.autograd as autograd 

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
# calculate epsilon


device = torch.device("cpu")

USE_CUDA = False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)



# prioritized 
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

def update_epsilon(n):
	epsilon_start = 1.0
	epsilon = epsilon_start
	epsilon_final = 0.4
	epsilon_decay = 1000 # N from the research paper (equation #6)

	epsilon = 1.0 - (n/epsilon_decay)

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
	    	print('not random')
	    	#observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
	    	actions = self.forward(observation)
	    	action = torch.argmax(actions).item()

	    else:
	        action = np.random.choice(env.action_space.n)

	    return action

replay_buffer = NaivePrioritizedBuffer(500000)

current_model = DQN().to(device)
target_model  = DQN().to(device)

optimizer = optim.Adam(current_model.parameters())

gamma = 0.99 # discount factor as mentioned in the paper

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

# TD loss
def compute_td_loss(batch_size):
    state, action, reward, next_state, done, indices = replay_buffer.sample(batch_size, 0.4) 


    Q_targets_next = target_model(next_state).detach().max(1)[0].unsqueeze(1)
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

def plot(frame_idx, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig('dqeaf.png')


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


def test_model():

	total_reward = 0
	F = 100 #total test files
	T = 80 # total mutations allowed
	ratio = F * 0.5 # if number of mutations generated is half the total size
	success = 0
	rn = RangeNormalize(-0.5,0.5)
	fe = pefeatures.PEFeatureExtractor()
	malware_path = "TestData\malware\*"
	all_files = glob.glob(malware_path)
	for episode in range(1, F):
		file_path = random.choice(all_files)
		state = env.reset(file_path)
		state_norm = rn(state)
		state_norm = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)
		for mutation in range(1, T):

			actions = current_model.forward(state_norm)
			print(actions)

			action = torch.argmax(actions).item()
			next_state, reward, done, _ = env.step(action)
			print('episode : ' + str(episode))
			print('mutation : ' + str(mutation))
			print('test action : ' + str(action))
			print('test reward : ' + str(reward))
			state = next_state
			state_norm = rn(state)
			state_norm = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)

			if(done):
				success = success + 1
				break

		if success >= ratio:
			print('success : ' + str(success))
			return True

	print('success : ' + str(success))
	return False



if __name__ == "__main__":

	D = 30000 # as mentioned in the research paper (total number of episodes)
	T = 80 # as mentioned in the paper (total number of mutations that the agent can perform on one file)
	B = 1000 # as mentioned in the paper (number of steps before learning starts)
	batch_size = 32 # as mentioned in the paper (batch_size)
	losses = []
	reward_ben = 20
	n = 0 #current training step
	rn = RangeNormalize(-0.5,0.5)
	check = False
	malware_path = "Data\malware\*"
	all_files = glob.glob(malware_path)
	fig, axs = plt.subplots(1,figsize=(10,8))  # Create a figure and a set of subplots
	for episode in range(1, D):
		file_path = random.choice(all_files)
		print("Training: ", file_path)
		state = env.reset(file_path)
		#print(state[0:7])
		state_norm = rn(state)
		state_norm = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)
		print('state')
		#print(state_norm[0:7])
		for mutation in range(1, T):
			n = n + 1
			epsilon = update_epsilon(n)
			action = current_model.chooseAction(state_norm, epsilon)
			print('episode : ' + str(episode))
			print('mutation : ' + str(mutation))
			print('action : ' + str(action))
			next_state, reward, done, _ = env.step(action)
			print('reward : ' + str(reward))
			next_state_norm = rn(next_state) 
			next_state_norm = torch.from_numpy(next_state_norm).float().unsqueeze(0).to(device)

			if reward == 10.0:
				power = -((mutation-1)/T)
				reward = (math.pow(reward_ben, power))*100

			replay_buffer.push(state_norm, action, reward, next_state_norm, done)

			if len(replay_buffer) > B:
				loss = compute_td_loss(batch_size)
				losses.append(loss.item())
				print('loss avg : ' + str(mean(losses)))


			if done:
				break

			state_norm = next_state_norm
			#print(state_norm[0:7])

		print("episode is over : ")
		print(episode)
		print(reward)

		if n % 100 == 0:
			print('updating target')
			update_target(current_model, target_model)
		rl_save_model_interval = 100
		rl_output_directory = "modified/updatedEpsilon/dqeaf"
		if episode % rl_save_model_interval == 0:
			if not os.path.exists(rl_output_directory):
				os.mkdir(rl_output_directory)
				print("[*] model directory has been created at : " + str(rl_output_directory))
			torch.save(current_model.state_dict(), os.path.join(rl_output_directory, "rl-model-" + str(episode) + "-" +str(date.today()) + ".pt" ))
			print("[*] Saving model in models/ directory ...")
		axs.clear()
		axs.plot(losses, color='b')  # Plot the losses on the second graph
		axs.set_ylabel('Losses', color='b')  # Label the y-axis of the second graph
		plt.pause(0.01)

		if episode % 550 == 0:
			print('testing model')
			check = test_model()
				
		if check:
			plt.savefig( "rl-model-" + str(episode) + "-" +str(date.today())+".png")
			break

	torch.save(current_model.state_dict(), 'dqeaf.pt')
	plot(episode, losses)
	


























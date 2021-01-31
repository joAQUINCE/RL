import gym
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class function_approximator(nn.Module):
    def __init__(self, features_in_cnt, hidden_layer_size, output_size):
        super(function_approximator, self).__init__()
        self.linear1 = nn.Linear(features_in_cnt, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_state):
        input_state = F.relu(self.linear1(input_state))
        input_state = self.linear2(input_state)
        return F.softmax(input_state, dim=1)
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        foward_softmax = self.forward(state).cpu()
        discrete_dist = Categorical(foward_softmax)
        sampled_action = discrete_dist.sample()
        return sampled_action.item(), discrete_dist.log_prob(sampled_action)

pi_star = function_approximator(6,32,3).to(device)

env = gym.make("Acrobot-v1")
env.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_list = []

def REINFORCE(episode_cnt, max_steps):
    print("Starting REINFORCE Routine")
    render_list = []
    lr = 0.001
    optimizer = optim.Adam(pi_star.parameters(), lr=lr, eps = 10**-6)
    rewards_episodes = []
     
    for episode_current in range(0, episode_cnt):
        state = env.reset()
        log_prob_list = []
        rewards = []

        if episode_current == 4000:
            optimizer = optim.Adam(pi_star.parameters(), lr=lr/2, eps = 10**-6)
        
        for step_current in range(max_steps):
            
            if episode_current > episode_cnt - 5:
                render_list.append(env.render(mode="rgb_array"))

            action, action_log_prob = pi_star.get_action(state)
            #acrobot actions are -1,0,1
            action = action -1
            
            log_prob_list.append(action_log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if done:
                break 

        rewards_episodes.append(sum(rewards))
        current_episode_reward = rewards_episodes[-1]

        policy_loss = []
        for action_log_prob in log_prob_list:
            policy_loss.append(-action_log_prob * current_episode_reward)
        policy_loss = torch.cat(policy_loss).sum()
        loss_list.append(policy_loss.detach().numpy())
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode_current % 100 == 0:
            print("Current Episode:", episode_current+1)
            print("Loss Under Current Policy", policy_loss)
            print("Average Score for Past 50 Episodes:", np.mean(rewards_episodes[-50:]),"\n")      
    
    env.close()    

    if len(render_list) >0:
        directory = "acrobot_gifs_REINFORCE/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        imageio.mimsave(directory+"lr_"+str(lr)+"_"+str(time.time())+".gif", render_list, fps=10)
 
    return rewards_episodes

rewards_episodes = REINFORCE(5000,1000)
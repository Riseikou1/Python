import numpy as np
import torch 
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import gymnasium as gym
from torch.distributions import MultiVariateNormal

hidden_dim = 256
lr_actor = 3e-4
lr_critic = 1e-3
gamma = 0.99
gae_lambda = 0.95
ppo_epochs = 10
mini_batch_size =64
ppo_clip = 0.2
buffer_size = 2048
update_timestep = buffer_size
action_std = 0.5

class ActorCritic(nn.Module):
    def __init__(self , num_actions):
        super(ActorCritic,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.fc1 = nn.Linear(64*7*7,hidden_dim)

        self.fc_actor = nn.Linear(hidden_dim,num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        self.fc_critic = nn.Linear(hidden_dim,1)
    def forward(self,state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        action_mean = self.fc_actor(x)
        action_var = torch.exp(self.log_std.expand_as(action_mean))
        cov_mat = torch.diag_embed(action_var)
        value = self.fc_critic(x)
        return action_mean,cov_mat,value
    
class Memory :
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.states =[]
        self.is_terminated = []
    def clear_memory(self) :
        del self.actions[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.states[:] 
        del self.is_terminated[:]

class PPO :
    def __init__(self,num_actions):
        self.policy = ActorCritic(num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(),lr = lr_actor)
        self.policy_old = ActorCritic(num_actions)
        self.policy_old.load_state_dict(self.policy.state.dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self,state):
        with torch.no_grad():
            action_mean,action_var,_ = self.policy_old(state)
            dist = MultiVariateNormal(action_mean,action_var)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.detach().numpy(),action_logprob.detach()
    def update(self,memory):
        rewards = list()
        discounted_reward = 0
        for reward,is_terminal in zip(reversed(memory.rewards),reversed(memory.is_terminals)):
            if is_terminal :
                discounted_reward = 0
            discounted_reward = reward + (gamma*discounted_reward)
            rewards.insert(0,discounted_reward)
        rewards = torch.tensor(rewards,dtype=torch.float32)
        rewards = (rewards - rewards.mean())/ (rewards.std()+1e-7)
        old_states = torch.stack(memory.states).detach()
        old_action = torch.stack(memory.states).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        for _ in range(ppo_epochs):
            

    
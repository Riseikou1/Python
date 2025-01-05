import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

state_dim = env.observatoin_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.99
tau = 0.005
buffer_size = 1e6
batch_size = 128
alpha = 0.2

class ReplayBuffer :
    def __init__(self,capacity) :
        self.buffer = deque(maxlen = int(capacity))
    def push(self,state,action,reward,next_state,done)  :
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size) :
        state,action,reward,next_state,done = zip(*random.sample(self.buffer,batch_size))
        return np.array(state),np.array(action),np.array(reward,dtype=np.float32),np.array(next_state),np.array(done,dtype=np.float32)
    def __len__(self) :
        return len(self.buffer)
    
class Actor(nn.Module) :
    def __init__(self) :
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.mean = nn.Linear(hidden_dim,action_dim)
        self.log_std = nn.Linear(hidden_dim,action_dim)
    def forward(self,state) :
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(log_std,min=-20,max=2)
        return mean,log_std
    def sample(self,state) :
        mean , log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean,std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        return action
    
class Critic(nn.Module) :
    def __init__(self) :
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
    def forward(self,state,action) :
        x = torch.cat([state,action],dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SACAgent :
    def __init__(self) :
        self.actor = Actor()
        self.critic_1 = Critic()
        self.critic_2 = Critic()
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr=critic_lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
    def select_action(self,state) :
        state = torch.FloatTensor(state).unsquueze(0)
        action = self.actor.sample(state)
        return action.detach().numpy()[0]
    def update(self,batch_size,gamma=gamma,tau=tau,alpha=alpha) :
        if len(self.replay_buffer) < batch_size :
            return
        state,action,reward,next_state,done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsquueze(1)
        with torch.no_grad() :
            next_state_action = self.actor.sample(next_state)
            target_q1_next = self.target_critic_1(next_state,next_state_action)
            target_q2_next = self.target_critic_2(next_state,next_state_action)
            target_q_min = torch.min(target_q1_next,target_q2_next)-alpha*torch.log(1-next_state-action.pow(2)+1e6)
            target_q = reward + (1-done)*gamma*target_q_min
        current_q1 = self.critic_1(state,action)
        critic_1_loss = F.mse_loss(current_q1,target_q)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_loss.backward()
        


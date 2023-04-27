import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model import Actor, ContActor, Dynamics
from .updates import vpg_update
from torch.distributions import Categorical
from .gaussian_model import hard_update
import copy

class VPG(nn.Module):
    def __init__(self, state_space, action_space, hidden_sizes=(64,64), activation=nn.Tanh, 
        alpha=3e-4, beta=3e-4, gamma=0.9, device="cpu", action_std=0.5, with_model=False):
        super(VPG, self).__init__()
        
        # deal with 1d state input
        state_dim = state_space.shape[0]
        
        self.gamma = gamma
        self.device = device
        self.with_model = with_model
        
        if isinstance(action_space, Discrete):
            self.discrete_action = True
            self.action_dim = action_space.n
            #new policy is for meta learning, every few iters, we update policy to new_policy
            self.new_policy = Actor(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            self.policy = Actor(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            self.policy_m = Actor(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            self.policy_m.load_state_dict(copy.deepcopy(self.policy.state_dict()))
            self.new_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))
            if with_model:
                self.model = Dynamics(state_dim, 1, hidden_sizes, activation, self.device).to(self.device)

        elif isinstance(action_space, Box):
            self.discrete_action = False
            self.action_dim = action_space.shape[0]
            self.policy = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.policy_m = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.policy_m.load_state_dict(copy.deepcopy(self.policy.state_dict()))
            if with_model:
                self.model = Dynamics(state_dim, self.action_dim, hidden_sizes, activation, self.device).to(self.device)

        self.state_dim = state_dim
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.action_std = action_std
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.optimizer_m = optim.Adam(self.policy_m.parameters(), lr=alpha)
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=beta)

        if with_model:
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.mse_loss = nn.MSELoss()
            self.bce_loss = nn.BCELoss()
    
    # def act_policy(self, state):
    #     return self.policy.act(state, self.device)

    def act_policy_m(self, state):
        return self.policy_m.act(state, self.device)

    def initialize_policy_m(self):
        self.policy_m.load_state_dict(copy.deepcopy(self.policy.state_dict()))

    # def update_policy(self, memory):
    #     discounted_reward = []
    #     Gt = 0
    #     for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
    #         if is_terminal:
    #             Gt = 0
    #         Gt = reward + (self.gamma * Gt)
    #         discounted_reward.insert(0, Gt)
    #
    #     # Normalizing the rewards:
    #     #        rewards = torch.tensor(rewards).to(self.device)
    #     #        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    #
    #     policy_gradient = []
    #     for log_prob, Gt in zip(memory.logprobs, discounted_reward):
    #         policy_gradient.append(-log_prob * Gt)
    #
    #     self.optimizer.zero_grad()
    #     policy_gradient = torch.stack(policy_gradient).sum()
    #     policy_gradient.backward()
    #     self.optimizer.step()
    
    def update_policy(self, memory):
        policy_m_para_before = copy.deepcopy(self.policy_m.state_dict())
        self.update_policy_m(memory)
        policy_m_para_after = copy.deepcopy(self.policy_m.state_dict())
        for key, meta_para in zip(policy_m_para_before, self.new_policy.parameters()):
            meta_para.data.copy_(meta_para.data +
                                 (policy_m_para_after[key]-policy_m_para_before[key])/self.alpha*self.beta)

        # vpg_update(self.optimizer, memory.logprobs, memory.rewards, memory.is_terminals, self.gamma)


        
#        rewards = []
#        discounted_reward = 0
#        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
#            if is_terminal:
#                discounted_reward = 0
#            discounted_reward = reward + (self.gamma * discounted_reward)
#            rewards.insert(0, discounted_reward)
#        
#        # Normalizing the rewards:
##        rewards = torch.tensor(rewards).to(self.device)
##        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
#        
#        policy_gradient = []
#        for log_prob, Gt in zip(memory.logprobs, rewards):
#            policy_gradient.append(-log_prob * Gt)
#        
#        self.optimizer.zero_grad()
#        policy_gradient = torch.stack(policy_gradient).sum()
#        policy_gradient.backward()
#        self.optimizer.step()

    def update_policy_m(self, memory):
        # caculate policy gradient
        discounted_reward = []
        Gt = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.gamma * Gt)
            discounted_reward.insert(0, Gt)

        policy_gradient = []
        gamma_pow = 1

        for log_prob, Gt, is_terminal in zip(memory.logprobs, discounted_reward, memory.is_terminals):
            policy_gradient.append(-log_prob * Gt * gamma_pow)
            if is_terminal:
                gamma_pow = 1
            else:
                gamma_pow *= self.gamma

        self.optimizer_m.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer_m.step()
        # create policy_m
        # if isinstance(self.action_space, Discrete):
        #     policy_m = Actor(self.state_dim, self.action_dim, self.hidden_sizes, self.activation, with_clone=True,
        #                      prior=self.policy.action_layer, lr=self.lr).to(self.device)
        # elif isinstance(self.action_space, Box):
        #     policy_m = ContActor(self.state_dim, self.action_dim, self.hidden_sizes, self.activation, self.action_std,
        #                          self.device, with_clone=True, prior=self.policy.action_layer, lr = self.lr).to(self.device)

        # # apply gradient descent
        # for layer, layer_m in zip(self.policy.action_layer, policy_m.action_layer):
        #     if type(layer) == nn.Linear:
        #         layer_m.weight = (layer_m.weight - self.lr * layer.weight.grad).clone()
        #         layer_m.bias = (layer_m.bias - self.lr * layer.bias.grad).clone()

        # return policy_m

    def update_model(self, op_memory, batchsize=256):
        states, actions, rewards, next_states, dones = op_memory.sample(batchsize)
        # print("state", states)
        # print("action", actions)
        # print("rewards", rewards)
        # print("dones", 1*dones)
        if self.discrete_action:
            actions = np.array([actions]).transpose()

        pred_delta, pred_rewards, pred_dones = self.model.predict(states, actions) 
        # print("preddone", pred_dones.flatten())
        # print("dones", 1*dones)
        loss = self.mse_loss(pred_rewards.flatten(), torch.tensor(rewards).float()) \
                + self.mse_loss(pred_delta+torch.tensor(states).float(), torch.tensor(next_states).float()) \
                + self.bce_loss(pred_dones.flatten(), torch.tensor(1*dones).float())
        
        self.model_optimizer.zero_grad()
        # print("model loss:", loss.item())
        loss.backward()
        self.model_optimizer.step()
        return loss.item()
    
    # def get_state_dict(self):
    #     return self.policy.state_dict(), self.optimizer.state_dict()
    
    
    # def set_state_dict(self, state_dict, optim):
    #     self.policy.load_state_dict(state_dict)
    #     self.optimizer.load_state_dict(optim)
    
    def set_params(self, sample_policy):
        for layer, sample_layer in zip(self.policy.action_layer, sample_policy.action_layer):
            # print(type(layer), type(sample_layer))
            if type(layer) == nn.Linear:
                # print("layer.weight", layer.weight)
                # print("sample_layer.weight", sample_layer.weight)
                hard_update(layer.weight, sample_layer.weight)
                hard_update(layer.bias, sample_layer.bias)
                # print("layer.weight", layer.weight)
                # print("sample_layer.weight", sample_layer.weight)

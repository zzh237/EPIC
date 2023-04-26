import copy
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
from .updates import ppo_update
from torch.distributions import Categorical
from .gaussian_model import hard_update

class PPO(nn.Module):
    def __init__(self, state_space, action_space, K_epochs=4, eps_clip=0.2, hidden_sizes=(64,64), 
                 activation=nn.Tanh, learning_rate=3e-4, gamma=0.9, device="cpu", action_std=0.5,
                 grad_clip_norm=7.0, iter_per_round=5):
        super(PPO, self).__init__()
        state_dim = state_space.shape[0]
        self.state_dim = state_dim
        self.action_space = action_space
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.lr = learning_rate

        self.gamma = gamma
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.iter_per_round = iter_per_round
        self.action_std = action_std

        if isinstance(action_space, Discrete):
            self.discrete_action = True
            self.action_dim = action_space.n
            self.new_policy = Actor(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            self.old_policy = Actor(state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            self.old_policy.action_layer.load_state_dict(copy.deepcopy(self.new_policy.action_layer.state_dict()))
        elif isinstance(action_space, Box):
            self.discrete_action = False
            self.action_dim = action_space.shape[0]
            self.new_policy = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.old_policy = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.old_policy.action_layer.load_state_dict(copy.deepcopy(self.new_policy.action_layer.state_dict()))

        self.optimizer = torch.optim.Adam(self.new_policy.action_layer.parameters(), lr=learning_rate)

    def act(self, state):
        return self.new_policy.act(state, self.device)

    def update_policy(self, memory):
        
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        
        ppo_update(self.new_policy, self.optimizer, old_logprobs, memory.rewards,
                   memory, self.gamma, self.K_epochs, self.eps_clip, self.loss_fn, self.device)

    def update_policy_m(self, memory):
        # calculate discounted reward
        discounted_reward = []
        Gt = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.gamma * Gt)
            discounted_reward.insert(0, Gt)

        for _ in range(self.iter_per_round):
            # calculate ratio of policy probabilities
            states_tensor = torch.stack(memory.states)
            actions_tensor = torch.stack(memory.actions).reshape((-1,len(memory.actions)))
            new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.new_policy, states_tensor, actions_tensor)
            old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.old_policy, states_tensor, actions_tensor)
            ratio_of_policy_probabilities = torch.exp(new_policy_distribution_log_prob) / (torch.exp(old_policy_distribution_log_prob) + 1e-8)
            #calculate loss and update parameters of a local policy
            ratio_of_policy_probabilities = torch.clamp(input=ratio_of_policy_probabilities,
                                                        min=-sys.maxsize,
                                                        max=sys.maxsize)
            discounted_reward_tensor = torch.tensor(discounted_reward).to(ratio_of_policy_probabilities)
            potential_loss_value_1 = discounted_reward_tensor * ratio_of_policy_probabilities
            potential_loss_value_2 = discounted_reward_tensor * torch.clamp(input=ratio_of_policy_probabilities,
                                                                            min=1.0 - self.eps_clip,
                                                                            max=1.0 + self.eps_clip)
            loss = torch.min(potential_loss_value_1, potential_loss_value_2)
            loss = -torch.mean(loss)
            # print('loss:{}'.format(loss.item()))
            #update parameters of policy for single task

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.new_policy.action_layer.parameters(),
                                           self.grad_clip_norm)  # clip gradients to help stabilise training
            self.optimizer.step()


        # create policy_m
        if isinstance(self.action_space, Discrete):
            policy_m = Actor(self.state_dim, self.action_dim, self.hidden_sizes, self.activation, with_clone=True,
                             prior=self.new_policy.action_layer, lr=self.lr).to(self.device)
        elif isinstance(self.action_space, Box):
            policy_m = ContActor(self.state_dim, self.action_dim, self.hidden_sizes, self.activation, self.action_std,
                                 self.device, with_clone=True, prior=self.new_policy.action_layer, lr=self.lr).to(self.device)
        return policy_m

    def equalize_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.old_policy.parameters(), self.new_policy.parameters()):
            old_param.data.copy_(new_param.data)

    def calculate_log_probability_of_actions(self, policy, states_tensor, actions_tensor):
        action_probs = policy.action_layer(states_tensor)
        policy_distribution = Categorical(action_probs)
        policy_distribution_log_prob = policy_distribution.log_prob(actions_tensor)
        return policy_distribution_log_prob

    def get_state_dict(self):
        return self.policy.state_dict(), self.optimizer.state_dict()
    
    def set_state_dict(self, state_dict, optim):
        self.policy.load_state_dict(state_dict)
        self.optimizer.load_state_dict(optim)

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
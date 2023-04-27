import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from .model import GaussianActor, GaussianContActor
from .updates import vpg_update
from torch.distributions import Categorical
from .gaussian_model import hard_update
import copy
from algos.agents.model import EltwiseLayer

def calculate_KL(mu1, sigma1, mu2, sigma2):
    '''

    Args:
        mu1: mean of default dist
        sigma1: std of default dist
        mu2: mean of prior dist
        sigma2: std of prior dist

    Returns:

    '''
    first_term = torch.sum(torch.log(sigma2/sigma1)) - len(sigma1)
    second_term = torch.sum(sigma1/sigma2)
    third_term = torch.sum((mu2-mu1).pow(2)/sigma2)

    return 0.5*(first_term+second_term+third_term)


class GaussianVPG(nn.Module):
    def __init__(self, state_space, action_space, mu, sigma, hidden_sizes=(64, 64), activation=nn.Tanh,
                 alpha=3e-4, beta=3e-4, gamma=0.9, device="cpu", action_std=0.5, lam=0.9, lam_decay=0.999):
        super(GaussianVPG, self).__init__()
        state_dim = state_space.shape[0]

        self.gamma = gamma
        self.device = device
        self.lam = lam
        self.lam_decay = lam_decay
        # self.with_model = with_model

        if isinstance(action_space, Discrete):
            self.discrete_action = True
            self.action_dim = action_space.n
            # new policy is for meta learning, every few iters, we update policy to new_policy
            self.new_default_policy = GaussianActor(state_dim, self.action_dim, hidden_sizes, activation, mu, sigma).to(
                self.device)
            self.default_policy = GaussianActor(state_dim, self.action_dim, hidden_sizes, activation, mu, sigma).to(self.device)
            self.prior_policy = GaussianActor(state_dim, self.action_dim, hidden_sizes, activation, mu, sigma).to(self.device)
            self.policy_m = GaussianActor(state_dim, self.action_dim, hidden_sizes, activation, mu, sigma).to(
                self.device)

            self.policy_m.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
            self.new_default_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
            self.prior_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))

        elif isinstance(action_space, Box):
            self.discrete_action = False
            self.action_dim = action_space.shape[0]

            self.new_default_policy = GaussianContActor(state_dim, self.action_dim, hidden_sizes, activation,
                                                action_std, mu, sigma, device).to(self.device)

            self.default_policy = GaussianContActor(state_dim, self.action_dim, hidden_sizes, activation,
                                                action_std, mu, sigma, device).to(self.device)

            self.prior_policy = GaussianContActor(state_dim, self.action_dim, hidden_sizes, activation,
                                            action_std, mu, sigma, device).to(self.device)

            self.policy_m = GaussianContActor(state_dim, self.action_dim, hidden_sizes, activation,
                                              action_std, mu, sigma, device).to(self.device)

            self.policy_m.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
            self.prior_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
            self.new_default_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))

        self.state_dim = state_dim
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.action_std = action_std
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.optimizer_m = optim.Adam(self.policy_m.parameters(), lr=alpha)
        self.optimizer = optim.Adam(self.policy_m.parameters(), lr=beta)

    def act_policy_m(self, state):
        return self.policy_m.act(state, self.device)

    def initialize_policy_m(self):
        self.policy_m.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
        for layer in self.policy_m.action_layer:
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                nn.init.constant_(layer.bias, val=0.0)
        # for policy_layer, policy_m_layer in zip(self.policy., self.policy_m.parameters()):
        # policy_m_param.data.copy_(policy_param.data)

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

    def update_policy_m_with_regularizer(self, memory, N):
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

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()

        # calculate regularizer
        reg = torch.tensor(0,dtype=torch.float32)
        for default_layer, prior_layer in zip(self.default_policy.action_layer, self.prior_policy.action_layer):
            assert type(default_layer)==type(prior_layer), "default_layer and prior_layer should match each other"
            if isinstance(default_layer,EltwiseLayer):
                reg += calculate_KL(mu1=default_layer.bias, sigma1=default_layer.weight,
                                    mu2=prior_layer.bias, sigma2=prior_layer.weight)
        reg = torch.sqrt((reg + torch.log(2*np.sqrt(torch.tensor(N))/0.01))/(2*N))
        # calculate total loss and back propagate
        total_loss = policy_gradient + reg
        total_loss.backward()

        self.optimizer.step()

    def update_mu_theta_for_default(self, memory, N):
        policy_m_para_before = copy.deepcopy(self.policy_m.state_dict())
        self.update_policy_m_with_regularizer(memory, N)
        policy_m_para_after = copy.deepcopy(self.policy_m.state_dict())
        for key, meta_para in zip(policy_m_para_before, self.new_default_policy.parameters()):
            meta_para.data.copy_(meta_para.data +
                                 (policy_m_para_after[key] - policy_m_para_before[key]))

    def update_default_and_prior_policy(self):
        # update prior distribution
        for prior_param, new_default_param in zip(self.prior_policy.parameters(), self.new_default_policy.parameters()):
            prior_param.data.copy_(self.lam *new_default_param.data + (1.0-self.lam)*prior_param.data)
        # update default distribution
        self.default_policy.load_state_dict(copy.deepcopy(self.new_default_policy.state_dict()))
        self.lam *= self.lam*self.lam_decay


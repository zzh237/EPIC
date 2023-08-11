import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from torch.distributions import Categorical, MultivariateNormal
import copy
import math


def calculate_KL(mu1, sigma1, mu2, sigma2):
    '''

    Args:
        mu1: mean of default dist
        sigma1: std of default dist
        mu2: mean of prior dist
        sigma2: std of prior dist

    Returns:

    '''
    first_term = torch.sum(torch.log(sigma2 / sigma1)) - len(sigma1)
    second_term = torch.sum(sigma1 / sigma2)
    third_term = torch.sum((mu2 - mu1).pow(2) / sigma2)

    return 0.5 * (first_term + second_term + third_term)


# -------------------------------------------------------------------------------------------
#  Stochastic linear layer and its support functions !START!
# -------------------------------------------------------------------------------------------
def get_param(shape):
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.FloatTensor(*shape))


def init_stochastic_linear(m, log_var_init):
    if log_var_init is None:
        log_var_init = {'mean': -10, 'std': 0.1}
    n = m.w_mu.size(1)
    stdv = math.sqrt(1. / n)
    m.w_mu.data.uniform_(-stdv, stdv)
    if m.use_bias:
        m.b_mu.data.uniform_(-stdv, stdv)
        m.b_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])
    m.w_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])


class StochasticLinear(nn.Module):

    def __init__(self, in_dim, out_dim, use_bias=True, prm_log_var_init=None):
        super(StochasticLinear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        weights_size = (out_dim, in_dim)
        self.use_bias = use_bias
        if use_bias:
            bias_size = out_dim
        else:
            bias_size = None
        self.create_stochastic_layer(weights_size, bias_size)
        init_stochastic_linear(self, prm_log_var_init)
        self.eps_std = 1.0

    def create_stochastic_layer(self, weights_size, bias_size):
        # create the layer parameters
        # values initialization is done later
        self.w_mu = get_param(weights_size)
        self.w_log_var = get_param(weights_size)
        # self.w = {'mean': self.w_mu}
        if bias_size is not None:
            self.b_mu = get_param(bias_size)
            self.b_log_var = get_param(bias_size)
            # self.b = {'mean': self.b_mu}

    def forward(self, x):
        # Layer computations (based on "Variational Dropout and the Local
        # Reparameterization Trick", Kingma et.al 2015)
        # self.operation should be linear or conv

        if self.use_bias:
            b_var = torch.exp(self.b_log_var)
            bias_mean = self.b_mu
        else:
            b_var = None
            bias_mean = None

        out_mean = self.operation(x, self.w_mu, bias=bias_mean)

        eps_std = self.eps_std
        if eps_std == 0.0:
            layer_out = out_mean
        else:
            w_var = torch.exp(self.w_log_var)
            out_var = self.operation(x.pow(2), w_var, bias=b_var)

            # Draw Gaussian random noise, N(0, eps_std) in the size of the
            # layer output:
            noise = out_mean.data.new(out_mean.size()).normal_(0, eps_std)
            # noise = randn_gpu(size=out_mean.size(), mean=0, std=eps_std)

            noise = Variable(noise, requires_grad=False)

            out_var = F.relu(out_var)  # to avoid nan due to numerical errors in sqrt
            layer_out = out_mean + noise * torch.sqrt(out_var)

        return layer_out

    def set_eps_std(self, eps_std):
        old_eps_std = self.eps_std
        self.eps_std = eps_std
        return old_eps_std

    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def operation(self, x, weight, bias):
        return F.linear(x, weight, bias)


class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation,device):
        super(GaussianActor, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        layers = []
        sizes = [state_dim] + hid + [action_dim]
        output_activation = nn.Softmax(dim=-1)
        for j in range(len(sizes) - 1):
            act = activation() if j < len(sizes) - 2 else output_activation
            layers += [StochasticLinear(sizes[j], sizes[j + 1]), act]

        self.action_layer = nn.Sequential(*layers).to(device)
        self.device = device

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return state, action, dist.log_prob(action)

    def act_prob(self, state, action, device):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)

        return action_logprobs

    def get_dist(self, state, device):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        return dist


class GaussianContActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_std, device):
        super(GaussianContActor, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        layers = []
        sizes = [state_dim] + hid + [action_dim]
        output_activation = nn.Tanh()
        for j in range(len(sizes) - 1):
            act = activation() if j < len(sizes) - 2 else output_activation
            layers += [StochasticLinear(sizes[j], sizes[j + 1]), act]

        self.action_layer = nn.Sequential(*layers).to(device)
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)
        self.device = device

    def act(self, state):
        if type(state) is tuple:
            state = state[0]
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(self.device)
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return state, action.detach(), action_logprob

    def act_prob(self, state, action):
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)

        return action_logprobs

    def get_dist(self, state):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().to(self.device)
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        return dist
# -------------------------------------------------------------------------------------------
#  Stochastic linear layer and its support functions !END!
# -------------------------------------------------------------------------------------------


class GaussianVPGMC(nn.Module):
    def __init__(self, state_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh,
                 alpha=3e-4, beta=3e-4, gamma=0.9, device="cpu", action_std=0.5, \
                    lam=0.9, lam_decay=0.999, m = 10):
        super(GaussianVPGMC, self).__init__()
        state_dim = state_space.shape[0]
        print("device is {}".format(device))
        self.gamma = gamma
        self.device = device
        self.lam = lam
        self.lam_decay = lam_decay
        self.m = m
        # self.with_model = with_model
        if isinstance(action_space, Discrete):
            self.discrete_action = True
            self.action_dim = action_space.n

            # new policy is for meta learning, every few iters, we update policy to new_policy
            self.new_default_policy = GaussianActor(state_dim, self.action_dim, hidden_sizes, activation,device).to(
                self.device)
            self.default_policy = GaussianActor(state_dim, self.action_dim, hidden_sizes, activation,device).to(
                self.device)
            self.prior_policy = GaussianActor(state_dim, self.action_dim, hidden_sizes, activation,device).to(
                self.device)
            self.policy_m = {j: GaussianActor(state_dim, self.action_dim, hidden_sizes, activation,device).to(
                self.device) for j in range(m)}
            for j in range(m):
                self.policy_m[j].load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
            self.new_default_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
            self.prior_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))

        elif isinstance(action_space, Box):
            self.discrete_action = False
            self.action_dim = action_space.shape[0]

            self.new_default_policy = GaussianContActor(state_dim, self.action_dim, hidden_sizes, activation,
                                                        action_std, device).to(self.device)

            self.default_policy = GaussianContActor(state_dim, self.action_dim, hidden_sizes, activation,
                                                    action_std, device).to(self.device)

            self.prior_policy = GaussianContActor(state_dim, self.action_dim, hidden_sizes, activation,
                                                  action_std, device).to(self.device)

            self.policy_m = {j: GaussianContActor(state_dim, self.action_dim, hidden_sizes, activation,
                                                  action_std, device).to(self.device) for j in range(m)}

            for j in range(m):
                self.policy_m[j].load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
            self.prior_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))
            self.new_default_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))

        self.state_dim = state_dim
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.action_std = action_std
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.optimizer_m = {j:optim.Adam(self.policy_m[j].parameters(), lr=alpha) for j in range(m)}
        self.optimizer = {j:optim.Adam(self.policy_m[j].parameters(), lr=beta) for j in range(m)}

    def act_policy_m(self, state, j):
        return self.policy_m[j].act(state)

    def initialize_policy_m(self):
        for j in range(self.m):
            self.policy_m[j].load_state_dict(copy.deepcopy(self.default_policy.state_dict()))

    def update_policy_m(self, memory, j):
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

        self.optimizer_m[j].zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer_m[j].step()

    def update_policy_m_with_regularizer(self, memories, N, H, j):
        memory = memories[j]
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

        policy_gradient = torch.stack(policy_gradient).sum()

        # calculate regularizer
        KL = []
        for policy_layer, prior_layer in zip(self.policy_m[j].action_layer, \
                                             self.prior_policy.action_layer):
            assert type(policy_layer) == type(prior_layer), "default_layer and prior_layer should match each other"
            if isinstance(policy_layer, StochasticLinear):
                KL.append(calculate_KL(mu1=policy_layer.w_mu, \
                                       sigma1=policy_layer.w_log_var,
                                        mu2=prior_layer.w_mu, \
                                            sigma2=prior_layer.w_log_var))

                KL.append(calculate_KL(mu1=policy_layer.b_mu, \
                                       sigma1=policy_layer.b_log_var,
                                       mu2=prior_layer.b_mu, \
                                        sigma2=prior_layer.b_log_var))
        KL = torch.stack(KL).sum()

        c = torch.tensor(1.5)
        delta = torch.tensor(0.01)
        epsilon = torch.log(torch.tensor(2.0))/(2*torch.log(c)) * \
            (1+torch.log(KL/np.log(2/delta)))
        reg = (1+c)/2*torch.sqrt(torch.tensor(2.0)) * \
            torch.sqrt((KL + np.log(2/delta) + epsilon) * N * H**2)

        # reg = torch.sqrt((KL + torch.log(2 * np.sqrt(torch.tensor(N)) / 0.01)) / (2*N))
        # reg = torch.sqrt(reg/(2*N))
        # calculate total loss and back propagate
        total_loss = policy_gradient + reg 
        self.optimizer[j].zero_grad()
        total_loss.backward()
        self.optimizer[j].step()

    def update_mu_theta_for_default(self, memories, N, H):
        v = {}
        for j in range(self.m):
            policy_m_para_before = copy.deepcopy(self.policy_m[j].state_dict())
            self.update_policy_m_with_regularizer(memories, N, H, j)
            # self.update_policy_m(memory)
            policy_m_para_after = copy.deepcopy(self.policy_m[j].state_dict())
            for key in policy_m_para_before:
                if j == 0:
                    v[key] = policy_m_para_after[key] - policy_m_para_before[key]
                else:
                    v[key]+=policy_m_para_after[key] - policy_m_para_before[key]
            
        for key, meta_para in zip(v, self.new_default_policy.parameters()):
            meta_para.data.copy_(meta_para.data + 1.6*v[key]/self.m)
       

    def update_default_and_prior_policy(self):
        # update prior distribution
        # for prior_param, new_default_param in zip(self.prior_policy.parameters(), self.new_default_policy.parameters()):
        #     prior_param.data.copy_((1 - self.lam) * new_default_param.data + self.lam * prior_param.data)
        # update default distribution
        self.default_policy.load_state_dict(copy.deepcopy(\
            self.new_default_policy.state_dict()))
        # #update prior distribution
        # self.prior_policy.load_state_dict(copy.deepcopy(self.new_default_policy.state_dict()))
        for prior_param, new_default_param in \
            zip(self.prior_policy.parameters(), \
                self.new_default_policy.parameters()):
            prior_param.data.copy_((1-self.lam)*new_default_param.data \
                                   + self.lam*prior_param.data)

        self.lam *= self.lam_decay

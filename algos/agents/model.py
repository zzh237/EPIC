import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
from algos.agents.gaussian_model import CloneLinear
import numpy

def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation() if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]

    return nn.Sequential(*layers)

def clone_mlp(prior, sizes, activation, output_activation=nn.Identity(), lr = -1):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation() if j < len(sizes) - 2 else output_activation
        layers += [CloneLinear(sizes[j], sizes[j + 1], prior[j*2], lr = lr), act]

    return nn.Sequential(*layers)

class Dynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, device):
        super(Dynamics, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        self.device = device
        self.mu = mlp([state_dim+action_dim] + hid + [state_dim], activation)
        # self.sigma = mlp([state_dim+action_dim] + hid + [state_dim], activation, nn.Tanh())
        self.reward = mlp([state_dim+action_dim] + hid + [1], activation)
        # self.normal = torch.distributions.Normal(0,1)
        self.done = mlp([state_dim] + hid + [1], activation, nn.Sigmoid())

    def predict(self, state, action):
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        cat = torch.cat([state, action],1)
        mu = self.mu(cat)
        # sigma = self.sigma(torch.cat([state, action],1))
        reward = self.reward(cat)
        # epsilon = self.normal.sample(state.size()).to(self.device)
        # sampled = mu + sigma * epsilon
        done = self.done(mu)
        return mu, reward, done

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, with_clone = False, prior = [], lr = -1):
        super(Actor, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        if not with_clone:
            self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Softmax(dim=-1))
        else:
            self.action_layer = clone_mlp(prior, [state_dim] + hid + [action_dim], activation, nn.Softmax(dim=-1), lr = lr)
        
        
    def act(self, state, device):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return state, action, dist.log_prob(action)
    
    def act_prob(self, state, action, device):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        
        return action_logprobs
#        state = torch.from_numpy(state).float().to(device) 
#        action_probs = self.action_layer(state)
#        
#        return action_probs[action]

    def get_dist(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        return dist

class ContActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_std, device, with_clone = False, prior = [], lr = -1):
        super(ContActor, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        if not with_clone:
            self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Tanh())
        else:
            self.action_layer = clone_mlp(prior, [state_dim] + hid + [action_dim], activation, nn.Tanh(), lr = lr)
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def act(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return state, action.detach(), action_logprob

    def act_prob(self, state, action, device):
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        
        return action_logprobs
    
    def get_dist(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        return dist


class Value(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        self.v_net = mlp([obs_dim] + hid + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.
    
class QValue(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        self.q = mlp([obs_dim + act_dim] + hid + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super(ActorCritic, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Softmax(dim=-1))
        
        # critic
        self.value_layer = mlp([state_dim] + hid + [1], activation)
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, device):
        state = torch.from_numpy(state).float().to(device) 
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
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        return dist
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
    
    
class ContActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_std, device):
        super(ContActorCritic, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
            
        # action mean range -1 to 1
        self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Tanh())
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        # critic
        self.value_layer = mlp([state_dim] + hid + [1], activation)
        
        self.device = device
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, device):
        state = torch.from_numpy(state).float().to(device) 
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        
        return state, action, dist.log_prob(action)
    
    def act_prob(self, state, action, device):
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        
        return action_logprobs
    
    def get_dist(self, state, device):
        if type(state) == numpy.ndarray:
            state = torch.from_numpy(state).float().to(device) 
        action_mean = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        return dist
    
    def evaluate(self, state, action):  
        action_mean = self.action_layer(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
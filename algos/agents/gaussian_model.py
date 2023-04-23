import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
import numpy
import math

def soft_update(target, source, tau):
    target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

def hard_update(target, source):
    target.data.copy_(source.data)

class Gaussian(object):
    def __init__(self, mu, rho, device):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
        self.device = device
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return (self.mu + self.sigma * epsilon).to(self.device)
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, device)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, device)

        self.weight_prior_mu = self.weight_mu.detach().clone()
        self.weight_prior_rho = self.weight_rho.detach().clone()
        self.weight_prior = Gaussian(self.weight_prior_mu, self.weight_prior_rho, device)
        # Bias parameters
        self.bias_prior_mu = self.bias_mu.detach().clone()
        self.bias_prior_rho = self.bias_rho.detach().clone()
        self.bias_prior = Gaussian(self.bias_prior_mu, self.bias_prior_rho, device)

        # self.weight_prior = Gaussian(self.weight_mu.detach().clone(),
        #     self.weight_rho.detach().clone(), self.device)
        # self.bias_prior = Gaussian(self.bias_mu.detach().clone(),
        #     self.bias_rho.detach().clone(), self.device)

        # Prior distributions
        # self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        # self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        # self.log_prior = 0
        # self.log_variational_posterior = 0

    # def forward(self, input, sample=False, calculate_log_probs=False):
    #     if self.training or sample:
    #         weight = self.weight.sample()
    #         bias = self.bias.sample()
    #     else:
    #         weight = self.weight.mu
    #         bias = self.bias.mu
    #     if self.training or calculate_log_probs:
    #         self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
    #         self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
    #     else:
    #         self.log_prior, self.log_variational_posterior = 0, 0

    #     return F.linear(input, weight, bias)

        
            

    def flatten_params(self):
        cur_mu = torch.cat((torch.flatten(self.weight.mu), torch.flatten(self.bias.mu)))
        prior_mu = torch.cat((torch.flatten(self.weight_prior.mu), torch.flatten(self.bias_prior.mu)))
        cur_sigma = torch.cat((torch.flatten(self.weight.sigma), torch.flatten(self.bias.sigma)))
        prior_sigma = torch.cat((torch.flatten(self.weight_prior.sigma), torch.flatten(self.bias_prior.sigma)))

        # p = torch.distributions.Normal(cur_mu, cur_sigma)
        # q = torch.distributions.Normal(prior_mu, prior_sigma)
        # loss = torch.distributions.kl_divergence(p, q).mean()
        # print(loss)
        return cur_mu, prior_mu, cur_sigma, prior_sigma

class SampleLinear(nn.Module):
    def __init__(self, in_features, out_features, prior: BayesianLinear):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = prior.weight.sample().clone()
        self.bias = prior.bias.sample().clone()
        self.weight.retain_grad()
        self.bias.retain_grad()
        # self.register_parameter('bias', self.bias)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class CloneLinear(nn.Module):
    def __init__(self, in_features, out_features, prior: nn.Linear, lr = -1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if lr == -1:
            self.weight = prior.weight.clone()
            self.bias = prior.bias.clone()
        else:
            self.weight = (prior.weight - lr * prior.weight.grad).clone()
            self.bias = (prior.bias - lr * prior.bias.grad).clone()
        self.weight.retain_grad()
        self.bias.retain_grad()

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation() if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]

    return nn.Sequential(*layers)

def gaussian_mlp(sizes, activation, output_activation=nn.Identity(), device="cpu"):
    layers = []
    for j in range(len(sizes)-1):
        layers.append(BayesianLinear(sizes[j], sizes[j+1], device))
        # act = activation() if j < len(sizes)-2 else output_activation
        # layers += [BayesianLinear(sizes[j], sizes[j+1], device), act]
    
    return layers # nn.Sequential(*layers)

def sample_mlp(prior, sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation() if j < len(sizes)-2 else output_activation
        layers += [SampleLinear(sizes[j], sizes[j+1], prior[j]), act]
    
    return nn.Sequential(*layers)

class PolicyHub(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, algo, tau, device):
        super(PolicyHub, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.tau = tau
        self.device = device

        if type(hidden_sizes) == int:
            self.hid = [hidden_sizes]
        else:
            self.hid = list(hidden_sizes)
        # actor
        self.gaussian_policy_layers = gaussian_mlp([self.state_dim] + self.hid + [self.action_dim], 
            self.activation, nn.Softmax(dim=-1), self.device)

        self.algo = algo 
        # for i, layer in enumerate(self.gaussian_policy_layers):
        #     print(i, "mu:", layer.weight_mu)
        #     print(i, "rho:", layer.weight_rho)
        #     print(i, "bmu:", layer.bias_mu)
        #     print(i, "brho:", layer.bias_rho)
            # print(i, "pri mu:", layer.weight_prior_mu)
            # print(i, "pri rho:", layer.weight_prior_rho)
            # print(i, "pri bmu:", layer.bias_prior_mu)
            # print(i, "pri brho:", layer.bias_prior_rho)

    def sample_weights(self):
        weights = []
        biases = []
        for layer in self.gaussian_policy_layers:
            weights.append(layer.weight.sample())
            biases.append(layer.bias.sample())
        return weights, biases
    
    def sample_policy(self, device):
        if self.algo in ['GaussianVPG']:
            out = SampleActor(self.gaussian_policy_layers, self.state_dim, self.action_dim,
            self.hidden_sizes, self.activation).to(device)
        if self.algo in ['GaussianPPO']:
            out = SampleActorCritic(self.gaussian_policy_layers, self.state_dim, self.action_dim,
            self.hidden_sizes, self.activation).to(device) 
        return out 
        # sample_mlp(self.gaussian_policy_layers, [self.state_dim] + self.hid + [self.action_dim], 
            # self.activation, nn.Softmax(dim=-1))
    
    def sample_cont_policy(self, action_std, device):
        if self.algo in ['GaussianVPG']:
            out = SampleContActor(self.gaussian_policy_layers, self.state_dim, self.action_dim,
                self.hidden_sizes, self.activation, action_std, device).to(device)
        if self.algo in ['GaussianPPO']:
            out = SampleContActorCritic(self.gaussian_policy_layers, self.state_dim, self.action_dim,
                self.hidden_sizes, self.activation, action_std, device).to(device)
        return out 
        
    def get_parameters(self):
        params = []
        for layer in self.gaussian_policy_layers:
            params.append(layer.weight_mu)
            params.append(layer.weight_rho)
            params.append(layer.bias_mu)
            params.append(layer.bias_rho)
        return params
    
    def regularize_loss(self):
        cur_mus = []
        prior_mus = []
        cur_sigmas = []
        prior_sigmas = []
        for layer in self.gaussian_policy_layers:
            cm, pm, cs, ps = layer.flatten_params()
            cur_mus.append(cm)
            prior_mus.append(pm)
            cur_sigmas.append(cs)
            prior_sigmas.append(ps)
        cur_mus = torch.cat(cur_mus)
        prior_mus = torch.cat(prior_mus)
        cur_sigmas = torch.cat(cur_sigmas)
        prior_sigmas = torch.cat(prior_sigmas)
        p = torch.distributions.Normal(cur_mus, cur_sigmas)
        q = torch.distributions.Normal(prior_mus, prior_sigmas)
        loss = torch.distributions.kl_divergence(p, q).mean().to(self.device)
        return loss

    def update_prior(self):
        # print("before soft update")
        # print(self.gaussian_policy_layers[0].bias_prior_mu)
        for layer in self.gaussian_policy_layers:
            soft_update(layer.weight_prior_mu, layer.weight_mu, self.tau)
            soft_update(layer.weight_prior_rho, layer.weight_rho, self.tau)
            soft_update(layer.bias_prior_mu, layer.bias_mu, self.tau)
            soft_update(layer.bias_prior_rho, layer.bias_rho, self.tau)
        # print("after soft update")
        # print(self.gaussian_policy_layers[0].bias_prior_mu)
    
    def load_params(self, source):
        for target_layer, source_layer in zip(self.gaussian_policy_layers, source.gaussian_policy_layers):
            hard_update(target_layer.weight_mu, source_layer.weight_mu)
            hard_update(target_layer.weight_rho, source_layer.weight_rho)
            hard_update(target_layer.bias_mu, source_layer.bias_mu)
            hard_update(target_layer.bias_rho, source_layer.bias_rho)

            hard_update(target_layer.weight_prior_mu, source_layer.weight_mu)
            hard_update(target_layer.weight_prior_rho, source_layer.weight_rho)
            hard_update(target_layer.bias_prior_mu, source_layer.bias_mu)
            hard_update(target_layer.bias_prior_rho, source_layer.bias_rho)

class SampleActor(nn.Module):
    def __init__(self, prior, state_dim, action_dim, hidden_sizes, activation):
        super(SampleActor, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        self.action_layer = sample_mlp(prior, [state_dim] + hid + [action_dim], activation, nn.Softmax(dim=-1))
        
        
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

    def get_parameters(self):
        params = []
        for layer in self.action_layer:
            if type(layer) == SampleLinear:
                params.append(layer.weight)
                params.append(layer.bias)
        return params

class SampleContActor(nn.Module):
    def __init__(self, prior, state_dim, action_dim, hidden_sizes, activation, action_std, device):
        super(SampleContActor, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        self.action_layer = sample_mlp(prior, [state_dim] + hid + [action_dim], activation, nn.Tanh())
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


class SampleActorCritic(nn.Module):
    def __init__(self, prior, state_dim, action_dim, hidden_sizes, activation):
        super(SampleActorCritic, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
        # actor
        self.action_layer = mlp([state_dim] + hid + [action_dim], activation, nn.Softmax(dim=-1))
        self.action_layer = sample_mlp(prior, [state_dim] + hid + [action_dim], activation, nn.Softmax(dim=-1))


        # critic
        self.value_layer = mlp([state_dim] + hid + [1], activation)
        
    # def forward(self):
    #     raise NotImplementedError
        
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
    
    
class SampleContActorCritic(nn.Module):
    def __init__(self, prior, state_dim, action_dim, hidden_sizes, activation, action_std, device):
        super(SampleContActorCritic, self).__init__()
        if type(hidden_sizes) == int:
            hid = [hidden_sizes]
        else:
            hid = list(hidden_sizes)
            
        # action mean range -1 to 1
        self.action_layer = sample_mlp(prior, [state_dim] + hid + [action_dim], activation, nn.Tanh())
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        # critic
        self.value_layer = mlp([state_dim] + hid + [1], activation)
        
        self.device = device
        
    # def forward(self):
    #     raise NotImplementedError
    
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
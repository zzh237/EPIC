"""
VPG compatible with the epic_mc_2 framework.
"""

from operator import itemgetter
from torch import nn
from torch import optim
import gym
from gym.spaces import Discrete, Box

from algos.memory import Memory
from algos.types import Action, EPICModel
from .model import Actor, ContActor, Dynamics
import copy
import torch

class VPG2(EPICModel):
    def __init__(self,
                 m: int,
                 env: gym.Env,
                 hidden_sizes: int | tuple[int, ...] = (64, 64),
                 activation = nn.Tanh,
                 lr=3e-4,
                 beta=3e-4,
                 discount=0.9,
                 device="cpu",
                 action_std=0.5,
                 with_model=False
                 ):
        super().__init__()

        assert env.observation_space.shape is not None

        state_dim = env.observation_space.shape[0]
        action_space = env.action_space
        self.device = device
        self._m = m
        self.state_dim = state_dim
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.action_std = action_std
        self.activation = activation
        self.lr = lr
        self.beta = beta
        self.discount = discount
        self.gamma = 1

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
            self.new_policy = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.policy = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.policy_m = ContActor(state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.policy_m.load_state_dict(copy.deepcopy(self.policy.state_dict()))
            self.new_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

            if with_model:
                self.model = Dynamics(state_dim, self.action_dim, hidden_sizes, activation, self.device).to(self.device)
        
        self.optimizer_m = optim.Adam(self.policy_m.parameters(), lr=lr)

        if with_model:
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.mse_loss = nn.MSELoss()
            self.bce_loss = nn.BCELoss()

        self.memory = Memory()

    @property
    def m(self):
        return self._m
    
    def act_m(self, m, state) -> Action:
        # state, action.detach(), action_logprob
        state, action, log_prob = self.policy_m.act(state, self.device)

        return Action(state=state, action=action, log_prob=log_prob)


    def per_step_m(self, m: int, meta_episode, step, action_dict: Action, reward, new_state, done):
        state, action, log_prob = itemgetter("state", "action", "log_prob")(action_dict)
        self.memory.add(state, action, log_prob, reward, done)

    def post_episode(self) -> None:
        # nothing to do after every episode
        pass

    def update_policy_m(self):
        # calculate policy gradient
        memory = self.memory
        discounted_reward = []
        Gt = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.discount * Gt)
            discounted_reward.insert(0, Gt)

        policy_gradient = []
        gamma_pow = 1

        for log_prob, Gt, is_terminal in zip(memory.logprobs, discounted_reward, memory.is_terminals):
            policy_gradient.append(-log_prob * Gt * gamma_pow)
            # many implementations of policy gradient do not include the discount, but it 
            # is included here for correctness
            if is_terminal:
                gamma_pow = 1
            else:
                gamma_pow *= self.discount

        policy_gradient = torch.stack(policy_gradient).sum()

        self.optimizer_m.zero_grad()
        policy_gradient.backward()
        self.optimizer_m.step()

    def post_meta_episode(self):
        # perform policy update via policy gradient
        policy_m_para_before = copy.deepcopy(self.policy_m.state_dict())
        self.update_policy_m()
        policy_m_para_after = copy.deepcopy(self.policy_m.state_dict())
        for key, meta_para in zip(policy_m_para_before, self.new_policy.parameters()):
            meta_para.data.copy_(meta_para.data +
                                 (policy_m_para_after[key]-policy_m_para_before[key])/self.lr*self.beta)
            

        self.memory.clear_memory()


    def update_prior(self) -> None:
        # this is a non-bayesian policy, so there is no prior to udpate
        pass

    def update_default(self) -> None:
        # non-bayesian policy, no default to update either
        pass

    def pre_meta_episode(self):
        # nothing to do
        pass

"""
Version of the non-bayesian PPO algorithm for the epic-mc-2 framework.
"""

from operator import itemgetter
from algos.memory import Memory
from algos.types import Action, EPICModel
import gym
from torch import nn
from gym.spaces import Discrete, Box
import copy
from .model import Actor, ContActor
import torch
import sys
from torch.distributions import Categorical


class PPO2(EPICModel):
    def __init__(self, m: int, env: gym.Env, K_epochs: int=4,
                 eps_clip:float = 0.2, hidden_sizes: int | tuple[int, ...] = (64, 64),
                 activation = nn.Tanh, lr: float = 3e-4, 
                 discount: float = 0.9,
                 device: str = "cpu",
                 action_std: float = 0.5,
                 grad_clip_norm = 7.0,
                 iter_per_round=5):
        super().__init__()
        assert env.observation_space.shape is not None
        self.state_dim = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.lr = lr
        self._m = m

        self.discount = discount
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.iter_per_round = iter_per_round
        self.action_std = action_std

        self.memory = Memory()

        if isinstance(env.action_space, Discrete):
            self.discrete_action = True
            self.action_dim = env.action_space.n
            self.new_policy = Actor(self.state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            self.old_policy = Actor(self.state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            # self.policy_m = Actor(self.state_dim, self.action_dim, hidden_sizes, activation).to(self.device)
            self.old_policy.action_layer.load_state_dict(copy.deepcopy(self.new_policy.action_layer.state_dict()))
        elif isinstance(env.action_space, Box):
            self.discrete_action = False
            self.action_dim = env.action_space.shape[0]
            self.new_policy = ContActor(self.state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.old_policy = ContActor(self.state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            # self.policy_m = ContActor(self.state_dim, self.action_dim, hidden_sizes, activation, action_std, self.device).to(self.device)
            self.old_policy.action_layer.load_state_dict(copy.deepcopy(self.new_policy.action_layer.state_dict()))

        self.optimizer = torch.optim.Adam(self.new_policy.action_layer.parameters(), lr=lr)
        self.to(device)

    @property
    def m(self) -> int:
        return self._m

    def act_m(self, m, state) -> Action:
        state, action, log_prob = self.old_policy.act(state, self.device)
        return Action(
            state=state,
            action=action,
            log_prob=log_prob
        )
    
    def per_step_m(self, m: int, meta_episode, step, action_dict: Action, reward, new_state, done):
        state, action, log_prob = itemgetter("state", "action", "log_prob")(action_dict)
        # add to memory
        self.memory.add(state, action, log_prob, reward, done)

    def update_policy_m(self, memory):
        # calculate discounted reward
        discounted_reward = []
        Gt = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                Gt = 0
            Gt = reward + (self.discount * Gt)
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

    def post_episode(self) -> None:
        # this algorithm updates after collecting enough episodes, but in post-meta-episode
        pass

    def update_prior(self) -> None:
        # no prior to update, this is a non-bayesian algorithm
        pass

    def update_default(self) -> None:
        # no default to update, this is a non-bayesian algorithm
        pass

    def pre_meta_episode(self):
        # nothing to do here
        pass

    def calculate_log_probability_of_actions(self, policy, states_tensor, actions_tensor):
        action_probs = policy.action_layer(states_tensor)
        policy_distribution = Categorical(probs=action_probs)
        policy_distribution_log_prob = policy_distribution.log_prob(actions_tensor)
        return policy_distribution_log_prob

    def post_meta_episode(self):
        # regular update.
        self.update_policy_m(self.memory)
        self.equalize_policies()


    def equalize_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.old_policy.parameters(), self.new_policy.parameters()):
            old_param.data.copy_(new_param.data)



    
import random
import numpy as np
from torch import from_numpy as tfn


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, as_tensors=False, device="cpu"):
        if batch_size > self.capacity:
            batch_size = self.capacity
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        if as_tensors:
            return( tfn(state).to(device), tfn(action).to(device), 
                   tfn(reward).to(device), tfn(next_state).to(device), 
                   tfn(done).to(device))
        else:
            return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)

    
    
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
    def add(self, state, action, logprob, reward, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
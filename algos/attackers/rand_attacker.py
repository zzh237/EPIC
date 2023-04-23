import sys
import torch  
import numpy as np  
import random

class RandAttacker:
    def __init__(self, radius=0.5, frac=1.0, maxat=1000):
        super(RandAttacker, self).__init__()

        self.radius = radius
        self.frac = frac
        self.maxat = maxat
        self.attack_num = 0
        
    def attack_r_general(self, memory):
        '''Attack with the current memory'''
        if self.attack_num >= self.maxat:
            print("exceeds budget")
            return memory.rewards
        randr = np.random.randn(len(memory.rewards))
        attack_r = self.proj(np.array(memory.rewards), randr, self.radius).tolist()
        
        if random.random() < self.frac:
            self.attack_num += 1
            return attack_r
        else:
            return memory.rewards
    
    
            
    def proj(self, old_r_array, new_r_array, radius):
        
        norm = np.linalg.norm(new_r_array-old_r_array)
        print("dist of r:", norm)
        proj_r = (old_r_array + (new_r_array - old_r_array) * radius / norm)
        return proj_r
import re
import math
import pandas as pd
import os
import itertools
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import numpy as np
import re 
import matplotlib.pyplot as plt


def smooth(scalars, weight=0.99):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def read_rewards(filename, samples=2000, episodes=10):
    rewards = []
    with open(filename, "r") as f:
        lines = f.readlines()
        last_line = lines[-1]
        x = re.split(': |, |\n', last_line)
        samples = int(x[1]) - 1
        for i in range(samples):
            rew_sum = 0
            for j in range(episodes):
                line = lines[i*episodes+j]
                rew = float(line.split()[-1])
                rew_sum += rew
            rewards.append(rew_sum / episodes)
    return rewards


def read_rewards_multi(filename, samples, episodes, runs):
    rewards = []
    for run in range(runs):
        reward = read_rewards(filename+"_run{}.txt".format(run), samples,episodes)
        rewards.append(smooth(reward))
    rewards = np.array(rewards)
    return np.mean(rewards, axis=0), np.std(rewards, axis=0)




if __name__ =="__main__":
    fig, ax = plt.subplots(figsize=(1.57 * 2, 1.18 * 2), dpi=600)
    for e in [1, 10]:
        epic_mean, epic_std = read_rewards_multi(filename='./results/test/multimodal/EPIC_CartPole-v0_vpg_s2000_n{}_every25_size32_c0.5_tau0.5_goal10.0_steps300_mass5.0'.format(e),
                                                samples=2000,
                                                episodes=e,
                                                runs=1)
    
    
        x_vals = list(range(len(epic_mean)))
        ax.plot(x_vals, epic_mean, label = e)
        ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1)
        ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1)
        ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1)

   

    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.legend(loc='best') 
    # plt.yticks([-15, -10, -5, 0])
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.show()
    plt.savefig('./results/test/multimodal/step300_epic25_episodescompare')

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
import random 
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


def read_rewards_mc(filename, samples=2000):
    rew = np.array([],dtype=float)
    std = np.array([],dtype=float) 
    with open(filename, "r") as f:
        lines = f.readlines()
        last_line = lines[-1]
        x = re.split(': |, |\n', last_line)
        samples = int(x[1])
        for i in range(samples):
            line = re.split(': |, |\n', lines[i])
            rew = np.append(rew, float(line[-4]))
            std = np.append(std, float(line[-2]))
    res = np.vstack((rew, std))
    return res

def read_rewards_multi_mc(filename, samples, runs):
    for run in range(runs):
        reward = read_rewards_mc(filename+"_run{}.txt".format(run), samples)
        reward = np_smooth(reward)
    return reward 

def np_smooth(targets, weight=0.99):  # Weight between 0 and 1
    final = []
    for i in range(2):
        scalars = targets[i,:] 
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = np.array([])
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed = np.append(smoothed, smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
        final.append(smoothed)
    final = np.array(final)
    return final.T


if __name__ =="__main__":
    fig, ax = plt.subplots(figsize=(1.57 * 2, 1.18 * 2), dpi=600)
    # for e in [1, 10]:
    #     epic_mean, epic_std = read_rewards_multi(filename='./results/test/multimodal/EPIC_CartPole-v0_vpg_s2000_n{}_every25_size32_c0.5_tau0.5_goal10.0_steps300_mass5.0'.format(e),
    #                                             samples=2000,
    #                                             episodes=e,
    #                                             runs=1)
    
    
    #     x_vals = list(range(len(epic_mean)))
    #     ax.plot(x_vals, epic_mean, label = e)
    #     ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1)
    #     ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1)
    #     ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1)

    for e in [10]:
        epic_mean, epic_std = read_rewards_multi(filename='./results/test/nosingle/multimodal/EPIC_CartPole-v0_vpg_s2000_n{}_every25_size32_c0.5_tau0.5_goal10.0_steps300_mass5.0'.format(e),
                                                samples=2000,
                                                episodes=e,
                                                runs=1)
    
    
        x_vals = list(range(len(epic_mean)))
        ax.plot(x_vals, epic_mean, label = "nosingle{}".format(e))
        ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1)
        ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1)
        ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1)

    colors = {}
    ms = [1]
    for i in ms:
        r = random.uniform(0, 1)
        g = random.uniform(0, 1)
        b = random.uniform(0, 1)
        colors[i] = (r,g,b)
    
    for m in ms:
        fname = './results/montecarlo/new/multimodal/EPIC_CartPole-v0_vpg_s2000_n10_every25_size32_c0.5_tau0.5_goal10.0_steps100_mass5.0_mc{}'.format(m)
        epic_mean_std = read_rewards_multi_mc(filename=fname,
                                             samples=2000,
                                             runs=1)
    
    
        x_vals = list(range(epic_mean_std.shape[0]))
        ax.plot(x_vals, epic_mean_std[:,0], color = colors[m], label = "mc{}".format(m))
        ax.plot(x_vals, epic_mean_std[:,0]+epic_mean_std[:,1],color = colors[m], alpha=0.5)
        ax.plot(x_vals, epic_mean_std[:,0]-epic_mean_std[:,1], color = colors[m],alpha=0.5)
        ax.fill_between(x_vals, y1=epic_mean_std[:,0]-epic_mean_std[:,1], \
                        y2=epic_mean_std[:,0]+epic_mean_std[:,1], color = colors[m],alpha=0.5)

    
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.legend(loc='best') 
    # plt.yticks([-15, -10, -5, 0])
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.show()
    plt.savefig('./results/montecarlo/multimodal/step300_epic25_episodes_nosingle_mc_compare')

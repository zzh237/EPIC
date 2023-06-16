import re
import math
import pandas as pd
import os
import itertools
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import numpy as np
import matplotlib.pyplot as plt


def smooth(scalars, weight=0.99):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def read_rewards(filename, samples=10, episodes=2000):
    rewards = []
    with open(filename, "r") as f:
        last_line = f.readlines()[-1]
        x = last_line.split()
        for i in range(samples):
            rew_sum = 0
            for j in range(episodes):
                line = f.readline()
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

    epic_mean, epic_std = read_rewards_multi(filename='./results/multimodal/EPIC_CartPole-v0_vpg_s2000_n10_every5_size32_c0.5_tau0.5_goal0.5_steps300_mass5.0',
                                             samples=2000,
                                             episodes=10,
                                             runs=1)
    # epic_mean = np.array(smooth(epic_mean, 0.99))
    # epic_std = 0.1 * np.array(smooth(epic_std, 0.99))

    maml_mean, maml_std = read_rewards_multi(filename='./results_maml/multimodal/maml_CartPole-v0_vpg_s2000_n10_every50_size32_goal0.5_steps300_mass5.0',
                                             samples=2000,
                                             episodes=10,
                                             runs=1)
    # maml_mean = np.array(smooth(maml_mean, 0.99))
    # maml_std = 0.1 * np.array(smooth(maml_std, 0.99))

    fig = plt.figure(figsize=(1.57 * 2, 1.18 * 2), dpi=600)

    x_vals = list(range(len(maml_mean)))
    plt.plot(x_vals, epic_mean, color='#D35400')
    plt.plot(x_vals, epic_mean+epic_std, color='#D35400', alpha=0.1)
    plt.plot(x_vals, epic_mean-epic_std, color='#D35400', alpha=0.1)
    plt.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1, color='#D35400')

    plt.plot(x_vals, maml_mean, color="#2980B9")
    plt.plot(x_vals, maml_mean+maml_std, color="#2980B9", alpha=0.1)
    plt.plot(x_vals, maml_mean-maml_std, color="#2980B9", alpha=0.1)
    plt.fill_between(x_vals, y1=maml_mean-maml_std, y2=maml_mean+maml_std, alpha=0.1, color="#2980B9")

    plt.xticks([0, 500, 1000, 1500, 2000])
    # plt.yticks([-15, -10, -5, 0])
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.show()
    plt.savefig('./figs_in_paper/multi_dyna_cartpole_step300_epic5_maml50')

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
import sys 


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
    if isinstance(runs, int):
        for run in range(runs):
            reward = read_rewards(filename+"_run{}.txt".format(run), samples,episodes)
            rewards.append(smooth(reward))
    else:
        reward = read_rewards(filename+"_run{}.txt".format(runs), samples,episodes)
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

def run_mc_plot():
    colors = {}
    ms = [1,2,5,10,15,20,50,100]
    rgb_1 = np.linspace(0,1,8)
    rgb_1 = list(itertools.permutations(rgb_1))[50]
    rgb_1 = [0,0.7,0,0.9,0.1,0.2,0.3,0.5]
    rgb_2 = np.linspace(1,0,8)
    rgb_2 = list(itertools.permutations(rgb_2))[10]
    rgb_2 = [0,0.7,0.4,0.4,0.6,0.2,0.3,0.5]
    rgb_3 = np.linspace(0,1,8)
    rgb_3 = list(itertools.permutations(rgb_3))[40]
    rgb_3 = [0,0.7,1,0.1,0.1,0.2,0.3,0.5]
    for j,i in enumerate(ms):
        r = random.uniform(0, 1)
        r = rgb_1[j]
        g = random.uniform(0, 1)
        g = rgb_2[j]
        b = random.uniform(0, 1)
        b = rgb_3[j]
        colors[i] = (r,g,b)
    steps = 300
    subfolder = 'new2'
    gradient = ''
    
    for name, ms_sep in zip(['1','2'],[ms[2:5],ms[5:8]]):
        fig, ax = plt.subplots(figsize=(1.57 * 2, 1.18 * 2), dpi=600)
        for e in [10]:
            epic_mean, epic_std = read_rewards_multi(filename='./results/test/nosingle_kl/multimodal/EPIC_CartPole-v0_vpg_s2000_n{}_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass5.0'.format(e,steps),
                                                    samples=2000,
                                                    episodes=e,
                                                    runs=1)
        
        
            x_vals = list(range(len(epic_mean)))
            ax.plot(x_vals, epic_mean, color='red',label = "reference")
            ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1)
            ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1)
            ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1)
        
        for m in ms_sep:
            fname = './results/montecarlo/{}/multimodal/EPIC_CartPole-v0_vpg_s2000_n10_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass5.0_mc{}'.format(subfolder,steps, m)
            epic_mean_std = read_rewards_multi_mc(filename=fname,
                                                samples=2000,
                                                runs=1)
            x_vals = list(range(epic_mean_std.shape[0]))
            ax.plot(x_vals, epic_mean_std[:,0], color = colors[m], label = r"Monte Carolo, $M=${}".format(m))
            ax.plot(x_vals, epic_mean_std[:,0]+epic_mean_std[:,1],color = colors[m], alpha=0.1)
            ax.plot(x_vals, epic_mean_std[:,0]-epic_mean_std[:,1], color = colors[m],alpha=0.1)
            ax.fill_between(x_vals, y1=epic_mean_std[:,0]-epic_mean_std[:,1], \
                            y2=epic_mean_std[:,0]+epic_mean_std[:,1], color = colors[m],alpha=0.1)

        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.legend(loc='upper left', labelspacing=0.5,fontsize=5) 
        # plt.yticks([-15, -10, -5, 0])
        # plt.tick_params(labelbottom=False, labelleft=False)
        # plt.show()
        plt.savefig('./results/montecarlo/{}/multimodal/step{}_epic25_episodes_nosingle_mc_compare_{}_{}'.format(subfolder, steps, name, gradient))

def run_mc_compare_maml_plot():
    colors = {}
    ms = [1,2,5,10,15,20,50,100]
    rgb_1 = np.linspace(0,1,8)
    rgb_1 = list(itertools.permutations(rgb_1))[50]
    rgb_1 = [0,0.7,0,0.9,0.1,0.2,0.3,0.5]
    rgb_2 = np.linspace(1,0,8)
    rgb_2 = list(itertools.permutations(rgb_2))[10]
    rgb_2 = [0,0.7,0.4,0.4,0.6,0.2,0.3,0.5]
    rgb_3 = np.linspace(0,1,8)
    rgb_3 = list(itertools.permutations(rgb_3))[40]
    rgb_3 = [0,0.7,1,0.1,0.1,0.2,0.3,0.5]
    for j,i in enumerate(ms):
        r = random.uniform(0, 1)
        r = rgb_1[j]
        g = random.uniform(0, 1)
        g = rgb_2[j]
        b = random.uniform(0, 1)
        b = rgb_3[j]
        colors[i] = (r,g,b)
    steps = 100
    subfolder = 'step100_5'
    name = 'maml'
    gradient = '1_35ave'
    mc_max = 10
    mc_compare = {}
    run_mc_key = False
    dynamics='multimodal'
    # dynamics='uniform'
    mass = 5.0
    goal = 0.5
    if dynamics == 'uniform':
        mass = 10.0
        goal=0.0
    if run_mc_key:
        for m in ms[2:]:
            fname = './results/montecarlo/{}/{}/EPIC_CartPole-v0_vpg_s2000_n10_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass{}_mc{}'.format(subfolder,dynamics,steps,mass, m)
            epic_mean_std = read_rewards_multi_mc(filename=fname,
                                                    samples=2000,
                                                    runs=1)
            x_vals = list(range(epic_mean_std.shape[0]))
            epic_mean = np.mean(epic_mean_std[:,0])
            mc_compare[m] = epic_mean
        mc_max = max(zip(mc_compare.values(), mc_compare.keys()))[1]
    

    for name, ms_sep in zip(['1','2'],[ms[2:5],ms[5:8]]):
        fig, ax = plt.subplots(figsize=(1.57 * 2, 1.18 * 2), dpi=600)
        # add the baselines
        for e in [10]:
            epic_mean, epic_std = read_rewards_multi(filename='./results/test/nosingle_kl/multimodal/EPIC_CartPole-v0_vpg_s2000_n{}_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass5.0'.format(e,steps),
                                                    samples=2000,
                                                    episodes=e,
                                                    runs=1)
        
        
            x_vals = list(range(len(epic_mean)))
            ax.plot(x_vals, epic_mean, color='red',label = "reference")
            ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1)
            ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1)
            ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1)
        
        # add the MAMLs
        N = 50 
        if dynamics == 'uniform':
            mass = 10.0
            goal=0.0
            N = 5
        maml_mean, maml_std = read_rewards_multi(filename='./results_maml/{}/maml_CartPole-v0_vpg_s2000_n10_every{}_size32_goal{}_steps{}_mass{}'.format(dynamics,N,goal,steps,mass),
                                             samples=2000,
                                             episodes=10,
                                             runs=4)
        x_vals = list(range(len(maml_mean)))
        lb = 'maml'
        ax.plot(x_vals, maml_mean, color = '#2980B9', label=lb)
        ax.plot(x_vals, maml_mean+maml_std, color = '#2980B9', alpha=0.1)
        ax.plot(x_vals, maml_mean-maml_std, color = '#2980B9', alpha=0.1)
        ax.fill_between(x_vals, y1=maml_mean-maml_std, y2=maml_mean+maml_std, alpha=0.1, color="#2980B9")
        

        for m in ms_sep:
            if m == mc_max:
                fname = './results/montecarlo/{}/{}/EPIC_CartPole-v0_vpg_s2000_n10_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass{}_mc{}'.format(subfolder,dynamics,steps, mass,m)
                epic_mean_std = read_rewards_multi_mc(filename=fname,
                                                    samples=2000,
                                                    runs=1)
                x_vals = list(range(epic_mean_std.shape[0]))
                ax.plot(x_vals, epic_mean_std[:,0], color = colors[10], label = r"EPICG".format(m))
                ax.plot(x_vals, epic_mean_std[:,0]+epic_mean_std[:,1],color = colors[10], alpha=0.1)
                ax.plot(x_vals, epic_mean_std[:,0]-epic_mean_std[:,1], color = colors[10],alpha=0.1)
                ax.fill_between(x_vals, y1=epic_mean_std[:,0]-epic_mean_std[:,1], \
                                y2=epic_mean_std[:,0]+epic_mean_std[:,1], color = colors[10],alpha=0.1)

        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.legend(loc='upper left', labelspacing=0.5,fontsize=5) 
        # plt.yticks([-15, -10, -5, 0])
        # plt.tick_params(labelbottom=False, labelleft=False)
        # plt.show()
        plt.savefig('./results/montecarlo/{}/{}/step{}_epic25_episodes_nosingle_mc_maml_compare_{}_{}'.format(subfolder, dynamics,steps, name, gradient))


def run_ablation():
    colors = {}
    ms = ["kl, no single", "kl, no single, N=1","kl, no single, using prior",\
          'single and kl', 'single, nokl','maml']
    rgb_1 = np.linspace(0,1,6)
    rgb_1 = list(itertools.permutations(rgb_1))[15]
    rgb_1 = [0,3.9,0.1,0.2,0.2,0.7]
    rgb_2 = np.linspace(1,0,6)
    rgb_2 = list(itertools.permutations(rgb_2))[10]
    rgb_2 = [0.7,0.4,0.6,0.2,0.5,0.7]
    rgb_3 = np.linspace(0,1,6)
    rgb_3 = list(itertools.permutations(rgb_3))[20]
    rgb_3 = [0.2,0.1,0.1,0.2,1,0.7]

    for j,i in enumerate(ms):
        r = random.uniform(0, 1)
        r = rgb_1[j]
        g = random.uniform(0, 1)
        g = rgb_2[j]
        b = random.uniform(0, 1)
        b = rgb_3[j]
        colors[i] = (r,g,b)

    subfolder = "test/nosingle_kl/default"
    steps = 100
    fig, ax = plt.subplots(figsize=(1.57 * 2, 1.18 * 2), dpi=600)
    for e in [10]:
        epic_mean, epic_std = read_rewards_multi(filename='./results/{}/multimodal/EPIC_CartPole-v0_vpg_s2000_n{}_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass5.0'.format(subfolder,e, steps),
                                                samples=2000,
                                                episodes=e,
                                                runs=5)
    
        lb = "kl, no single"
        label = "Add Regularizer"
        x_vals = list(range(len(epic_mean)))
        ax.plot(x_vals, epic_mean, label = label, color = colors[lb])
        ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1, color = colors[lb])
        ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1, color = colors[lb])
        ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1, color = colors[lb])
    
    
    # epic_mean, epic_std = read_rewards_multi(filename='./results/{}/multimodal/EPIC_CartPole-v0_vpg_s2000_n{}_every1_size32_c0.5_tau0.5_goal10.0_steps{}_mass5.0'.format(subfolder,e, steps),
    #                                             samples=2000,
    #                                             episodes=e,
    #                                             runs=3)
    
    # lb = "kl, no single, N=1"
    # x_vals = list(range(len(epic_mean)))
    # ax.plot(x_vals, epic_mean, label = lb, color = colors[lb])
    # ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1, color = colors[lb])
    # ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1, color = colors[lb])
    # ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1, color = colors[lb])
    


    
    # subfolder = "test/nosingle_kl/prior"
    # epic_mean, epic_std = read_rewards_multi(filename='./results/{}/multimodal/EPIC_CartPole-v0_vpg_s2000_n{}_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass5.0'.format(subfolder,e, steps),
    #                                             samples=2000,
    #                                             episodes=e,
    #                                             runs=5)
    
    
    # x_vals = list(range(len(epic_mean)))
    # lb = "kl, no single, using prior"
    # ax.plot(x_vals, epic_mean, label = lb, color = colors[lb])
    # ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1, color = colors[lb])
    # ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1, color = colors[lb])
    # ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1, color = colors[lb])
    
    # epic_mean, epic_std = read_rewards_multi(filename='./results/test/single/multimodal/EPIC_CartPole-v0_vpg_s2000_n10_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass5.0'.format(steps),
    #                                             samples=2000,
    #                                             episodes = 10,
    #                                             runs=4)
    
    # lb = 'single and kl'
    # x_vals = list(range(len(epic_mean)))
    # ax.plot(x_vals, epic_mean, label = lb, color = colors[lb])
    # ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1, color = colors[lb])
    # ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1, color = colors[lb])
    # ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1, color = colors[lb])
    

    subfolder = "test/single_nokl"
    epic_mean, epic_std = read_rewards_multi(filename='./results/{}/multimodal/EPIC_CartPole-v0_vpg_s2000_n10_every25_size32_c0.5_tau0.5_goal10.0_steps{}_mass5.0'.format(subfolder,steps),
                                                samples=2000,
                                                episodes = 10,
                                                runs=5)
    lb = 'single, nokl'
    label = 'No Regularizer'
    x_vals = list(range(len(epic_mean)))
    ax.plot(x_vals, epic_mean, label = label, color = colors[lb])
    ax.plot(x_vals, epic_mean+epic_std,  alpha=0.1,color = colors[lb])
    ax.plot(x_vals, epic_mean-epic_std,  alpha=0.1, color = colors[lb])
    ax.fill_between(x_vals, y1=epic_mean-epic_std, y2=epic_mean+epic_std, alpha=0.1, color = colors[lb])
    
    # maml_mean, maml_std = read_rewards_multi(filename='./results_maml/multimodal/maml_CartPole-v0_vpg_s2000_n10_every50_size32_goal0.5_steps{}_mass5.0'.format(steps),
    #                                          samples=2000,
    #                                          episodes=10,
    #                                          runs=1)
    # x_vals = list(range(len(maml_mean)))
    # lb = 'maml'
    # plt.plot(x_vals, maml_mean, color = colors[lb], label=lb)
    # plt.plot(x_vals, maml_mean+maml_std, color = colors[lb], alpha=0.1)
    # plt.plot(x_vals, maml_mean-maml_std, color = colors[lb], alpha=0.1)
    # plt.fill_between(x_vals, y1=maml_mean-maml_std, y2=maml_mean+maml_std, alpha=0.1, color="#2980B9")
    
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.legend(loc='upper left', labelspacing=0.5,fontsize=5) 
    # plt.yticks([-15, -10, -5, 0])
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.show()
    subfolder = 'test/single'
    plt.savefig('./results/{}/multimodal/step{}_epic25_episodes_single_kl_nokl_compare'.format(subfolder, steps))


if __name__ =="__main__":
    # run_mc_plot()
    run_mc_compare_maml_plot()
    # run_ablation()
    sys.exit(0)
    
    

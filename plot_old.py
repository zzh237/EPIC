import matplotlib.pyplot as plt
import numpy as np
import re
import math
import pandas as pd

def read_rewards(filename, samples, episodes):
    rewards = []
    with open(filename, "r") as f:
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
        rewards.append(reward)
    rewards = np.array(rewards)
    # print("rewards", rewards)
    return np.mean(rewards, axis=0), np.std(rewards, axis=0)

def read_rewards_multi_old(samples, episodes, coeff, runs, nometa=False):
    rewards = []
    for run in range(runs):
        if nometa:
            reward = read_rewards("results/CartPole-v0_vpg_s{}_n{}_goal0.5_c{}_nometa_run{}.txt".format(
            samples,episodes,coeff, run), samples,episodes)
        else:
            reward = read_rewards("results/CartPole-v0_vpg_s{}_n{}_goal0.5_c{}_run{}.txt".format(
                samples,episodes,coeff, run), samples,episodes)
        rewards.append(reward)
    rewards = np.array(rewards)
    # print("rewards", rewards)
    return np.mean(rewards, axis=0)

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


if __name__ == "__main__":
    n = 10
    s = 2000
    runs = 1
    xs = list(range(s))

    SMALL_SIZE = 20
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 25

    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams["figure.figsize"] = (20,10)

# for baselines
#     for tau in [0.8]:
#         for every in [50]:
#             res, std = read_rewards_multi("results/n50/Lunar_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
#             mu1 = np.array(smooth(res, 0.99))
#             sigma1=  0.1 * np.array(smooth(std, 0.99))
#             plt.plot(xs, mu1, color = 'b', label="PB-LRL")
#             plt.fill_between(xs,mu1+ sigma1, mu1-sigma1, color='b', alpha=0.1)

# #meta
#     # for tau in [0.8]:
#     #     for every in [50]:
#     #         res = read_rewards_multi("results/n50/Lunar_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
#     #         plt.plot(xs, smooth(res, 0.99), label="every" + str(every))

#     for tau in [0.8]:
#         for every in [50]:
#             res, std = read_rewards_multi("results/maml_Lunar_vpg_s{}_n{}_every{}_size32".format(s,n,every), s, n, runs)
#             mu1 = np.array(smooth(res, 0.99))
#             sigma1 = 0.1 * np.array(smooth(std, 0.99))
#             plt.plot(xs, mu1, label="MAML")
#             plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, alpha=0.1)


#     for tau in [0.8]:
#         for every in [50]:
#             res,std  = read_rewards_multi("results/no_meta/Lunar_vpg_s{}_n{}_every{}_size32_c0.5_tau{}_nometa".format(s,n,every,tau), s, n, runs)
#             mu1 = np.array(smooth(res, 0.99))
#             sigma1 = 0.1 * np.array(smooth(std, 0.99))
#             plt.plot(xs, mu1, label="Singe-task")
#             plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, alpha=0.1)

# #meta
#     for tau in [0.8]:
#         for every in [50]:
#             res, std= read_rewards_multi("results/n50/Lunar_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
#             mu1= np.array(smooth(res, 0.99))
#             sigma1=  np.array(smooth(std, 0.99))
#             plt.plot(xs, mu1, label="every" + str(every), color = 'b')
#             plt.fill_between(xs,mu1+ sigma1, mu1-sigma1, color='b', alpha=0.1)




    # res = read_rewards_multi("results/Lunar_vpg_s{}_n{}_c0.5_maml".format(s,n), s, n, runs)
    # plt.plot(xs, smooth(res, 0.99), label="every"+str(50))


##### Swimmer baseline comparison
    # dirname = "results_swimmer/"
    # s = 1000
    # runs = 10
    # xs = list(range(s))

    # for tau in [0.5]:
    #     for every in [25]:
    #         res, std = read_rewards_multi(dirname+"/Swimmer_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1=  0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color = 'b', label="PB-LRL")
    #         plt.fill_between(xs,mu1+ sigma1, mu1-sigma1, color='b', alpha=0.1)


    # for tau in [0.5]:
    #     for every in [25]:
    #         res, std = read_rewards_multi("results_swimmer_maml/Swimmer_vpg_s{}_n{}_every{}_size32".format(s,n,every), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1 = 0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color="#2ca02c", label="MAML")
    #         plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#2ca02c", alpha=0.1)

    # for tau in [0.5]:
    #     for every in [50]:
    #         res,std  = read_rewards_multi(dirname+"/Swimmer_vpg_s{}_n{}_every{}_size32_c0.5_tau{}_nometa".format(s,n,every,tau), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1 = 0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color="#ff7f0e", label="Singe-task")
    #         plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#ff7f0e", alpha=0.1)

    # plt.legend()
    # plt.xlabel("Tasks (environments)")
    # plt.ylabel("Mean reward")
    # # plt.show()
    # plt.savefig("plots/swimmer.png", format="png")
    # # tikzplotlib.save("plots/swimmer.tex")
    


##### Swimmer N comparison
    # dirname = "results_swimmer/"
    # s = 1000
    # runs = 10
    # xs = list(range(s))
    # linestyle = {10:"dashdot", 25: "dotted", 50: "solid", 75: "dashed"}
    # for tau in [0.5]:
    #     for every in [10,25,50]:
    #         res, std = read_rewards_multi(dirname+"/Swimmer_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1=  0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color = 'b', label="PB-LRL N=" + str(every), linestyle=linestyle[every])
    #         plt.fill_between(xs,mu1+ sigma1, mu1-sigma1, color='b', alpha=0.1)


    # for tau in [0.5]:
    #     for every in [50]:
    #         res,std  = read_rewards_multi(dirname+"/Swimmer_vpg_s{}_n{}_every{}_size32_c0.5_tau{}_nometa".format(s,n,every,tau), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1 = 0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color="#ff7f0e", label="Singe-task")
    #         plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#ff7f0e", alpha=0.1)

    # plt.legend()
    # plt.xlabel("Tasks (environments)")
    # plt.ylabel("Mean reward")
    # # plt.show()
    # plt.savefig("plots/swimmer_N.png", format="png")

    
    
    
    
    
    ##### cartpole baseline comparison
    dirname = "results/goal0.1"
    s = 2000
    runs = 10
    xs = list(range(s))

    # for tau in [0.5]:
    #     for every in [25]:
    #         # if every in [10,75]:
    #         #     res, std = read_rewards_multi(dirname+"/CartPole-v0_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
    #         # else:
    #         #     res, std = read_rewards_multi(dirname+"/CartPole-v0_vpg_s{}_n{}_every{}_goal0.5_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
    #         res, std = read_rewards_multi(dirname+"/CartPole-v0_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1=  0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color = 'b', label="PB-LRL")
    #         plt.fill_between(xs,mu1+ sigma1, mu1-sigma1, color='b', alpha=0.1)


    for tau in [0.5]:
        for every in [25]:
            res, std = read_rewards_multi("results/goal0.1/maml_CartPole-v0_vpg_s{}_n{}_every{}_size32".format(s,n,every), s, n, runs)
            mu1 = np.array(smooth(res, 0.99))
            sigma1 = 0.1 * np.array(smooth(std, 0.99))
            plt.plot(xs, mu1, color="#2ca02c", label="MAML")
            plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#2ca02c", alpha=0.1)

    # for tau in [0.5]:
    #     for every in [50]:
    #         res,std  = read_rewards_multi(dirname+"/CartPole-v0_vpg_s{}_n{}_every50_size32_c0.5_tau0.5_nometa".format(s,n), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1 = 0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color="#ff7f0e", label="Singe-task")
    #         plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#ff7f0e", alpha=0.1)

    plt.legend()
    plt.xlabel("Tasks (environments)")
    plt.ylabel("Mean reward")
    plt.show()
    # plt.savefig("plots/cart_goal0.1.png", format="png")
    # tikzplotlib.save("plots/swimmer.tex")
    
    
    
    
##### CartPole N comparison
    # dirname = "results"
    # s = 2000
    # runs = 10
    # xs = list(range(s))
    # linestyle = {10:"dashdot", 25: "dotted", 50: "solid", 75: "dashed"}
    # for tau in [0.8]:
    #     for every in [25,50,75]:
    #         if every in [10,75]:
    #             res, std = read_rewards_multi(dirname+"/CartPole-v0_vpg_s{}_n{}_every{}_size32_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
    #         else:
    #             res, std = read_rewards_multi(dirname+"/CartPole-v0_vpg_s{}_n{}_every{}_goal0.5_c0.5_tau{}".format(s,n,every,tau), s, n, runs)
            
    #         mu1 = np.array(smooth(res, 0.995))
    #         sigma1=  0.1 * np.array(smooth(std, 0.995))
    #         plt.plot(xs, mu1, color = 'b', label="PB-LRL N=" + str(every), linestyle=linestyle[every])
    #         plt.fill_between(xs,mu1+ sigma1, mu1-sigma1, color='b', alpha=0.1)


    # for tau in [0.8]:
    #     for every in [50]:
    #         res,std  = read_rewards_multi(dirname+"/CartPole-v0_vpg_s{}_n{}_goal0.5_c0.5_nometa".format(s,n), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.995))
    #         sigma1 = 0.1 * np.array(smooth(std, 0.995))
    #         plt.plot(xs, mu1, color="#ff7f0e", label="Singe-task")
    #         plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#ff7f0e", alpha=0.1)

    # plt.legend()
    # plt.xlabel("Tasks (environments)")
    # plt.ylabel("Mean reward")
    # # plt.show()
    # plt.savefig("plots/cart_N.png", format="png")
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import pandas as pd
import os
import itertools
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

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
    return np.mean(rewards, axis=0), np.std(rewards, axis=0)


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

# s, smaples, tasks; every, meta update; algo: algos, n: max episodes
class plot_all:
    
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

    def __init__(self, params,envsparams):
        envs = ['CartPole-v0','lunar','swimmer']
        # self.cartpole = envs['cartpole']
        # self.lunar = envs['lunar']
        # self.swimmer = envs['swimmer']
        self.envsparams = envsparams 
        self.params = {'algos':['epic','maml','single'], \
            'everys': None, \
                'ss':[2000], \
                    'ns':[1], \
                        'runs':1}

        self.params.update(params)
        self._plot() 
    
    def _plot(self):  
        
        algos = self.params['algos']
        everys = self.params['everys']
        ss = self.params['ss']
        ns = self.params['ns']
        runs = self.params['runs']
        
        def temp(d):
            out = ''
            for k, v in d.items(): 
                out+= '_{}{}'.format(k,v[0])
            return out  
        # for tau in [0.8]:
        def draw_meta(xs):
            res, std = read_rewards_multi(dname+"/results/soft_update_EPIC_{}_vpg_s{}_n{}_every{}_size32_c0.5_tau0.5".format(env, s,n,every)+ env_out, s, n, runs)
            mu1 = np.array(smooth(res, 0.99))
            sigma1=  0.1 * np.array(smooth(std, 0.99))
            plt.plot(xs, mu1, color = 'b', label="EPIC")
            plt.fill_between(xs,mu1+ sigma1, mu1-sigma1, color='b', alpha=0.1)
        def draw_maml(xs):
            res, std = read_rewards_multi(dname+"/results_maml/maml_meta_{}_vpg_s{}_n{}_every{}_size32".format(env,s,n,every) + env_out, s, n, runs)
            mu1 = np.array(smooth(res, 0.99))
            sigma1 = 0.1 * np.array(smooth(std, 0.99))
            plt.plot(xs, mu1, color="#2ca02c", label="MAML")
            plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#2ca02c", alpha=0.1)
        def draw_single(xs):
            res,std  = read_rewards_multi(dname+"/results/{}_vpg_s{}_n{}_every{}_size32_c0.5_tau0.5".format(env, s,n,every) + env_out, s, n, runs)
            mu1 = np.array(smooth(res, 0.99))
            sigma1 = 0.1 * np.array(smooth(std, 0.99))
            plt.plot(xs, mu1, color="#ff7f0e", label="Singe task")
            plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#ff7f0e", alpha=0.1)
        
        algodraws = {'meta': draw_meta, \
            'maml':draw_maml,\
                'single':draw_single}
        for env in envs:
            envparam = self.envsparams[env]
            if envparam is None:
                continue
            env_out = temp(envparam)
            for every in everys:
                for n in ns:
                    for s in ss:
                        xs = list(range(s))
                        for algo in algos:
                            algodraws[algo](xs)

            plt.legend()
            plt.xlabel("Tasks (environments)")
            plt.ylabel("Mean reward")
            plt.show()
            plt.savefig("plots/{}_{}.png".format(env, env_out), format="png")

if __name__ == "__main__":

    envs = {'CartPole-v0':{'goal':[0.5],'steps':[50], 'mass':[1.0]},\
        'lunar':None,\
            'swimmer':None} 
    params = {'algos':['meta','maml','single'], \
            'everys': [50], \
                'ss':[2000], \
                    'ns':[10], \
                        'runs':1}
    pl = plot_all(params, envs)
    

##### Lunar baseline comparison
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




#     res = read_rewards_multi("results/Lunar_vpg_s{}_n{}_c0.5_maml".format(s,n), s, n, runs)
#     plt.plot(xs, smooth(res, 0.99), label="every"+str(50))



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
    
    # s = 2000
    # runs =1
    # xs = list(range(s))

    # for tau in [0.5]:
    #     for every in [50]:
    #         res, std = read_rewards_multi(dname+"/results/meta_CartPole-v0_vpg_s{}_n{}_every{}_size32_c0.5_tau{}_goal0.5_steps50_mass1.0".format(s,n,every,tau), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1=  0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color = 'b', label="EPIC")
    #         plt.fill_between(xs,mu1+ sigma1, mu1-sigma1, color='b', alpha=0.1)


    # for tau in [0.5]:
    #     for every in [50]:
    #         res, std = read_rewards_multi(dname+"/results_maml/maml_meta_CartPole-v0_vpg_s{}_n{}_every{}_size32_steps50_mass1.0".format(s,n,every), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1 = 0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color="#2ca02c", label="MAML")
    #         plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#2ca02c", alpha=0.1)

    # for tau in [0.5]:
    #     for every in [50]:
    #         res,std  = read_rewards_multi(dname+"/results/CartPole-v0_vpg_s{}_n{}_every50_size32_c0.5_tau0.5_goal0.5_steps50_mass1.0".format(s,n), s, n, runs)
    #         mu1 = np.array(smooth(res, 0.99))
    #         sigma1 = 0.1 * np.array(smooth(std, 0.99))
    #         plt.plot(xs, mu1, color="#ff7f0e", label="Singe-task")
    #         plt.fill_between(xs, mu1 + sigma1, mu1 - sigma1, color="#ff7f0e", alpha=0.1)

    # plt.legend()
    # plt.xlabel("Tasks (environments)")
    # plt.ylabel("Mean reward")
    # plt.show()
    # plt.savefig("plots/cart_goal0.5_mass1.0.png", format="png")
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
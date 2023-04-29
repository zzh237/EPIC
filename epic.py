import torch
import torch.nn as nn
import argparse
import gym
import os 
import mujoco_py
import numpy as np
from gym.spaces import Box, Discrete
import setup
from algos.memory import Memory, ReplayMemory
from algos.agents.new_gaussian_vpg import GaussianVPG
from algos.agents.gaussian_ppo import GaussianPPO
from envs.new_cartpole import NewCartPoleEnv
from envs.new_lunar_lander import NewLunarLander
from envs.swimmer_rand_vel import SwimmerEnvRandVel
from envs.half_cheetah_rand_dir import HalfCheetahEnvRandDir
from envs.half_cheetah_rand_vel import HalfCheetahEnvRandVel
from envs.ant_rand_dir import AntEnvRandDir
from envs.ant_rand_goal import AntEnvRandGoal
from envs.ant_rand_vel import AntEnvRandVel
# from stable_baselines.common.env_checker import check_env

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
# change cpu to cuda if running on server
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--run', type=int, default=0)
# env settings
# Swimmer for majuco environment
parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--samples', type=int, default=2000) # need to tune
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--goal', type=float, default=0.5) 
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--mass', type=float, default=1.0) 
parser.add_argument('--action_std', type=float, default=0.5)

# meta settings
parser.add_argument('--meta', dest='meta', action='store_true')
parser.add_argument('--no-meta', dest='meta', action='store_false')
parser.set_defaults(meta=True)
parser.add_argument('--meta-episodes', type=int, default=10)  # need to tune
parser.add_argument('--coeff', type=float, default=0.5)  # need to tune
parser.add_argument('--tau', type=float, default=0.5)  # need to tune

# learner settings
parser.add_argument('--learner', type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--beta', type=float, default=1e-4)
# parser.add_argument('--update_every', type=int, default=300)
parser.add_argument('--meta_update_every', type=int, default=25)  # need to tune
parser.add_argument('--hiddens', nargs='+', type=int)
parser.add_argument('--lam', type=float, default=0.9)
parser.add_argument('--lam_decay', type=float, default=0.95)
parser.add_argument('--prm_log_var_init', type=float, default={'mean': -10, 'std': 0.1})


# file settings
parser.add_argument('--logdir', type=str, default="logs/")
parser.add_argument('--resdir', type=str, default="results/")
parser.add_argument('--moddir', type=str, default="models/")
parser.add_argument('--loadfile', type=str, default="")

args = parser.parse_args()

def get_log(file_name):
    logger = logging.getLogger('train') 
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(file_name, mode='a') 
    fh.setLevel(logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    return logger

def make_cart_env(seed, env="CartPole-v0"):
    # need to tune
    mass = 0.1 * np.random.randn() + args.mass 
    # print("a new env of mass:", mass)
    env = NewCartPoleEnv(masscart=mass)
    # goal = args.goal * np.random.randn() + 0.0
    # print("a new env of goal:", goal)
    # env = NewCartPoleEnv(goal=goal)
    # check_env(env, warn=True)
    return env

def make_lunar_env(seed, env="LunarLander-v2"):
    # need to tune
    # mass = 0.1 * np.random.randn() + 1.0
    # print("a new env of mass:", mass)
    # env = NewCartPoleEnv(masscart=mass)
    goal = np.random.uniform(-1, 1)
    # print("a new env of goal:", goal)
    env = NewLunarLander(goal=goal)
    # check_env(env, warn=True)
    return env

def make_car_env(seed, env="MountainCarContinuous-v0"):
    # need to tune
    env = gym.make("MountainCarContinuous-v0")
    return env

def make_mujoco_env(seed, env="Swimmer"):
    if env == "Swimmer":
        env = SwimmerEnvRandVel()
    elif env == "Halfcdir":
        env = HalfCheetahEnvRandDir()
    elif env == "Halfcvel":
        env = HalfCheetahEnvRandVel()
    elif env == "Antdir":
        env = AntEnvRandDir()
    elif env == "Antgol":
        env = AntEnvRandGoal()
    elif env == "Antvel":
        env = AntEnvRandVel()
#     check_env(env, warn=True)
    return env

envs = {'Swimmer':make_mujoco_env, 'LunarLander-v2': make_lunar_env, 'CartPole-v0':make_cart_env}

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env #"LunarLander-v2"
    # env_name = "LunarLander-v2"
    samples = args.samples
    max_episodes = args.episodes        # max training episodes
    max_steps = args.steps         # max timesteps in one episode
    meta_episodes = args.meta_episodes
    learner = args.learner

    alpha = args.alpha
    beta = args.beta
    device = args.device
    update_every = args.update_every
    meta_update_every = args.meta_update_every
    use_meta = args.meta
    coeff = args.coeff
    tau = args.tau
    action_std = args.action_std
    lam = args.lam
    lam_decay = args.lam_decay
    prm_log_var_init = args.prm_log_var_init
    ############ For All #########################
    gamma = 0.99                # discount factor
    render = False
    save_every = 100
    if args.hiddens:
        hidden_sizes = tuple(args.hiddens) # need to tune
    else:
        hidden_sizes = (32,32)
    activation = nn.Tanh  # need to tune

    use_model = False
    
    torch.cuda.empty_cache()
    ########## file related ####
    if env_name == 'Swimmer':
        filename = env_name + "_" + learner + "_s" + str(samples) + "_n" + str(max_episodes) \
            + "_every" + str(meta_update_every) \
                + "_size" + str(hidden_sizes[0]) + "_c" + str(coeff) + "_tau" + str(tau)\
                    + "_steps" + str(max_steps)
    else:
        filename = env_name + "_" + learner + "_s" + str(samples) + "_n" + str(max_episodes) \
            + "_every" + str(meta_update_every) + "_size" + str(hidden_sizes[0]) \
                + "_c" + str(coeff) + "_tau" + str(tau) \
                    + "_goal" + str(args.goal)\
                        + "_steps" + str(max_steps)\
                            + "_mass" + str(args.mass)
    if not use_meta:
        filename += "_nometa"

    if args.run >=0:
        filename += "_run" + str(args.run)
    
    if not os.path.exists(args.resdir):
        os.makedirs(args.resdir)  
    # rew_file = open(args.resdir + filename + ".txt", "w")
    meta_rew_file = open(args.resdir + "EPIC_" + filename + ".txt", "w")

    # env = gym.make(env_name)
    envfunc = envs[env_name]
    env = envfunc(args.seed, env_name)

    if learner == "vpg":
        print("-----initialize meta policy-------")
        actor_policy = GaussianVPG(env.observation_space, env.action_space,
                                  hidden_sizes=hidden_sizes, activation=activation, alpha=alpha,
                                  beta=beta, action_std=action_std, gamma=gamma, device=device,
                                  lam=lam, lam_decay=lam_decay, prm_log_var_init=prm_log_var_init)
    if learner == "ppo":
        print("-----initialize meta policy-------")
        # here we could also use PPO, need to check difference between them
        actor_policy = GaussianPPO(env.observation_space, env.action_space, meta_update_every,
                hidden_sizes=hidden_sizes, activation=activation, gamma=gamma, device=device, 
                learning_rate=lr, coeff=coeff, tau=tau)

        

    for sample in range(samples):
        meta_memory = Memory()
        memory = Memory()
        print("#### Learning environment sample {}".format(sample))
        ########## creating environment
        # env = gym.make(env_name)
        env = envfunc(args.seed, env_name)
        # env.seed(sample)

        ########## sample a meta learner
        actor_policy.initialize_policy_m() # initial policy theta
        # print("weight of layer 0", sample_policy.action_layer[0].weight) 

        #use single task policy to collect some trajectories
        start_episode = 0
        for episode in range(start_episode, max_episodes):
            state = env.reset()
            rewards = []
            for steps in range(max_steps):
                state_tensor, action_tensor, log_prob_tensor = actor_policy.act_policy_m(state)
                if isinstance(env.action_space, Discrete):
                    action = action_tensor.item()
                else:
                    action = action_tensor.cpu().data.numpy().flatten()
                new_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
                state = new_state
                if done or steps == max_steps-1:
                    # meta_rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                    #     sample, episode, np.round(np.sum(rewards), decimals = 3)))
                    break
        #update single task policy using the trajectory
        actor_policy.update_policy_m(memory)
        memory.clear_memory()
        #use updated single task policy to collect some trajectories
        for episode in range(start_episode, meta_episodes):
            state = env.reset()
            rewards = []
            for steps in range(max_steps):
                if render:
                    env.render()
                state_tensor, action_tensor, log_prob_tensor = actor_policy.act_policy_m(state)

                if isinstance(env.action_space, Discrete):
                    action = action_tensor.item()
                else:
                    action = action_tensor.cpu().data.numpy().flatten()
                new_state, reward, done, _ = env.step(action)

                rewards.append(reward)
                meta_memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
                state = new_state

                if done or steps == max_steps - 1:
                    meta_rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                        sample, episode, np.round(np.sum(rewards), decimals=3)))
                    break

        actor_policy.update_mu_theta_for_default(meta_memory, meta_update_every, sample)
        meta_memory.clear_memory()

        if (sample+1) % meta_update_every == 0:
            actor_policy.update_default_and_prior_policy()

        env.close()

    # rew_file.close()
    meta_rew_file.close()

            

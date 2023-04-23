import torch
import torch.nn as nn
import argparse
import gym
# import mujoco_py
import numpy as np
from gym.spaces import Box, Discrete
import setup
from algos.memory import Memory
from algos.agents.vpg import VPG
from algos.agents.ppo import PPO
from algos.agents.gaussian_vpg import GaussianVPG
from algos.agents.gaussian_model import PolicyHub
from envs.new_cartpole import NewCartPoleEnv
from envs.new_lunar_lander import NewLunarLander
from envs.swimmer_rand_vel import SwimmerEnvRandVel
from envs.half_cheetah_rand_dir import HalfCheetahEnvRandDir
# from stable_baselines.common.env_checker import check_env

import logging
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--run', type=int, default=-1)
# env settings Swimmer for majuto
parser.add_argument('--env', type=str, default="CartPole-v0")
parser.add_argument('--samples', type=int, default=2000) # need to tune
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--goal', type=float, default=0.5)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--mass', type=float, default=1.0) 

# meta settings
parser.add_argument('--meta', dest='meta', action='store_true')
parser.add_argument('--no-meta', dest='meta', action='store_false')
parser.set_defaults(meta=True)
parser.add_argument('--meta_episodes', type=int, default=10)  # need to tune
parser.add_argument('--coeff', type=float, default=0.5)  # need to tune
parser.add_argument('--tau', type=float, default=0.5)  # need to tune

# learner settings
parser.add_argument('--learner', type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--update_every', type=int, default=300)
parser.add_argument('--meta_update_every', type=int, default=50)  # need to tune
parser.add_argument('--hiddens', nargs='+', type=int)

# file settings
parser.add_argument('--logdir', type=str, default="logs/")
parser.add_argument('--resdir', type=str, default="results_maml/")
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


def make_cart_env(seed):
    # need to tune
    mass = 0.1 * np.random.randn() + args.mass 
    print("a new env of mass:", mass)
    env = NewCartPoleEnv(masscart=mass)
    # goal = args.goal * np.random.randn() + 0.0
    # print("a new env of goal:", goal)
    # env = NewCartPoleEnv(goal=goal)
    # check_env(env, warn=True)
    return env


def make_lunar_env(seed):
    # need to tune
    goal = np.random.uniform(-1, 1)
    print("a new env of goal:", goal)
    env = NewLunarLander(goal=goal)
    return env

def make_env(env, seed):
    if env == "CartPole-v0":
        env = make_cart_env(seed)
    elif env == "LunarLander-v2":
        env = make_lunar_env(seed)
    return env

def make_mujoco_env(env="Swimmer"):
    if env == "Swimmer":
        env = SwimmerEnvRandVel()
    elif env == "Antdir":
        env = HalfCheetahEnvRandDir()
#     check_env(env, warn=True)
    return env

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env  # "LunarLander-v2"
    env_name = "LunarLander-v2"
    samples = args.samples
    max_episodes = args.episodes  # max training episodes
    max_steps = args.steps  # max timesteps in one episode
    meta_episodes = args.meta_episodes
    learner = args.learner
    lr = args.lr
    device = args.device
    update_every = args.update_every
    meta_update_every = args.meta_update_every
    use_meta = args.meta
    coeff = args.coeff
    tau = args.tau
    ############ For All #########################
    gamma = 0.99  # discount factor
    render = False
    save_every = 100
    if args.hiddens:
        hidden_sizes = tuple(args.hiddens) # need to tune
    else:
        hidden_sizes = (32,32)
    activation = nn.Tanh  # need to tune

    torch.cuda.empty_cache()
    ########## file related
    filename = env_name + "_" + learner + "_s" + str(samples) + "_n" + str(max_episodes) \
        + "_every" + str(meta_update_every) \
        + "_size" + str(hidden_sizes[0]) \
            + "_mass" + str(args.mass) 

    if args.run >= 0:
        filename += "_run" + str(args.run)

    rew_file = open(args.resdir + "maml_" + filename + ".txt", "w")
    meta_rew_file = open(args.resdir + "maml_" + "meta_" + filename + ".txt", "w")

    # env = gym.make(env_name)
    # env = make_cart_env(args.seed)
	if env_name == "Swimmer":
		env = make_mujoco_env(env_name)
	else:
    	env = make_env(env_name, args.seed)

    if learner == "vpg":
        actor_policy = VPG(env.observation_space, env.action_space, hidden_sizes=hidden_sizes,
                           activation=activation, gamma=gamma, device=device, learning_rate=lr, with_meta=True)

    meta_memory = Memory()
    for sample in range(samples):
        print("sample " + str(sample))
        # env = make_cart_env(sample)
		if env_name == "Swimmer":
			env = make_mujoco_env(env_name)
		else:
			env = make_env(env_name, sample)
		

        memory = Memory()

        start_episode = 0
        timestep = 0

        for episode in range(start_episode, max_episodes):
            state = env.reset()
            rewards = []
            for steps in range(max_steps):
                if render:
                    env.render()

                state_tensor, action_tensor, log_prob_tensor = actor_policy.act(state)

                if isinstance(env.action_space, Discrete):
                    action = action_tensor.item()
                else:
                    action = action_tensor.cpu().data.numpy().flatten()
                new_state, reward, done, _ = env.step(action)

                rewards.append(reward)
                memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)

                state = new_state

                if done or steps == max_steps - 1:
                    rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                        sample, episode, np.round(np.sum(rewards), decimals=3)))
                    break

        # obtain policy_m and apple gradient descent
        policy_m = actor_policy.update_policy_m(memory)
        memory.clear_memory()

        # obtain meta_memory using updated policy_m
		if env_name == "Swimmer":
			meta_episodes = max_episodes
        for episode in range(start_episode, meta_episodes):
            state = env.reset()
            rewards = []
            for steps in range(max_steps):
                if render:
                    env.render()

                state_tensor, action_tensor, log_prob_tensor = policy_m.act(state, device)

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

        if (sample+1) % meta_update_every == 0:
            actor_policy.update_policy(meta_memory)
            meta_memory.clear_memory()

        env.close()

    rew_file.close()
    meta_rew_file.close()



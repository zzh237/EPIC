import torch
import torch.nn as nn
import argparse
import gym

import os 
import mujoco_py
import random
import numpy as np
from gym.spaces import Box, Discrete
import setup
from algos.memory import Memory, ReplayMemory
from algos.agents.new_gaussian_vpg import GaussianVPG
from algos.agents.gaussian_ppo import GaussianPPO
from envs.new_cartpole import NewCartPoleEnv
from envs.new_lunar_lander import NewLunarLander
from envs.new_swimmer import new_Swimmer
from envs.new_ant import new_AntEnv

from envs.new_half_cheetah import new_HalfCheetahEnv
from envs.new_walker_2d import new_Walker2dEnv
# from stable_baselines.common.env_checker import check_env

import logging
from datetime import datetime
## this is version 2.0

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
# change cpu to cuda if running on server
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--run', type=int, default=0)
# env settings
# Swimmer for majuco environment
parser.add_argument('--env', type=str, default="Ant",
parser.add_argument('--env', type=str, default="half_cheetah",
                    help=['Swimmer', 'LunarLander-v2', 'CartPole-v0', 'half_cheetah', 'Ant',
                          'walker_2d'])
parser.add_argument('--samples', type=int, default=2000) # need to tune
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--steps', type=int, default=100)
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
parser.add_argument('--update_every', type=int, default=300)
parser.add_argument('--meta_update_every', type=int, default=50)  # need to tune
parser.add_argument('--hiddens', nargs='+', type=int)
parser.add_argument('--lam', type=float, default=0.9)
parser.add_argument('--lam_decay', type=float, default=0.95)


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
    assert env=="CartPole-v0", "env_name should be CartPole-v0."
    if args.mass==5:
      masscart=np.random.choice(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), p=[0.15,0.18,0.34,0.18,0.15])
      masspole=np.random.choice(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), p=[0.34,0.18, 0.18, 0.15, 0.15])
      length=np.random.choice(np.array([0.3, 0.4, 0.5, 0.6, 0.7]), p=[0.15,0.18,0.34,0.18,0.15])
      masscart = 0.1 * np.random.randn() + masscart
      masspole = 0.01 * np.random.rand() + masspole
      length = 0.01*np.random.rand() + length
      env = NewCartPoleEnv(masscart=masscart,
                         masspole=masspole,
                         length=length)
    elif args.mass == 10:
      masscart = np.random.uniform(1, 5)
      masspole = np.random.uniform(0.1, 0.5)
      length = np.random.uniform(0.3, 0.7)
      env = NewCartPoleEnv(masscart=masscart,
                         masspole=masspole,
                         length=length)
    elif args.goal == 5:
      goalcart=np.random.choice(np.array([-0.99, -0.5, 0, 0.5, 0.99]), p=[0.15,0.18,0.34,0.18,0.15])
      goalcart = 0.1 * np.random.randn() + goalcart
      env = NewCartPoleEnv(goal=goalcart)
    elif args.goal == 10:
      goalcart=np.random.uniform(-1,1)
      env = NewCartPoleEnv(goal=goalcart)
    else:
      env = NewCartPoleEnv()
    return env

def make_lunar_env(seed, env="LunarLander-v2"):
    # goal = np.random.uniform(-1, 1)
    assert env=="LunarLander-v2"
    if args.mass == 5:
      main_engine_power = np.random.choice(np.array([11.0, 12.0, 13.0, 14.0, 15.0]),
                                         p=[0.15,0.18,0.34,0.18,0.15])
      side_engine_power = np.random.choice(np.array([0.45, 0.55, 0.65, 0.75, 0.85]),
                                         p=[0.15,0.18,0.34,0.18,0.15])
      main_engine_power = main_engine_power + 0.1*np.random.randn()
      side_engine_power = side_engine_power + 0.01*np.random.randn()
      env = NewLunarLander(main_engine_power=main_engine_power,
                         side_engine_power=side_engine_power)
    elif args.mass == 10:
      main_engine_power = np.random.uniform(3, 20)
      side_engine_power = np.random.uniform(0.15, 0.95)
      env = NewLunarLander(main_engine_power=main_engine_power,
                         side_engine_power=side_engine_power)
    elif args.goal == 5:
      goal=np.random.choice(np.array([-0.99, -0.5, 0, 0.5, 0.99]), p=[0.15,0.18,0.34,0.18,0.15])
      goal = 0.1 * np.random.randn() + goal
      env = NewLunarLander(goal=goal)
    elif args.goal == 10:
      goal=np.random.uniform(-1,1)
      env = NewLunarLander(goal=goal)
    else:
      env = NewLunarLander()
    # check_env(env, warn=True)
    return env

def make_car_env(env="MountainCarContinuous-v0"):
    # need to tune
    env = gym.make("MountainCarContinuous-v0")
    return env

# def make_mujoco_env(env="Swimmer"):
#     if env == "Swimmer":
#         # goal = np.random.uniform(0.1, 0.2)
#         # env = SwimmerEnvRandVel(goal=goal)
#         from gym.envs.mujoco.swimmer import SwimmerEnv
#         env = SwimmerEnv()
#     elif env == "Halfcdir":
#         env = HalfCheetahEnvRandDir()
#     elif env == "Halfcvel":
#         env = HalfCheetahEnvRandVel()
#     elif env == "Antdir":
#         env = AntEnvRandDir()
#     elif env == "Antgol":
#         env = AntEnvRandGoal()
#     elif env == "Antvel":
#         env = AntEnvRandVel()
# #     check_env(env, warn=True)
#     return env

def make_half_cheetah(env='half_cheetah'):
    assert env == 'half_cheetah', "env_name should be half_cheetah."
    env = new_HalfCheetahEnv()
    return env

def make_swimmer(env='Swimmer'):
    goal = np.random.uniform(low=-0.5, high=0.5)
    env = new_Swimmer(goal=goal)
    return env

def make_ant(env='Ant'):
    assert env=='Ant'
    env = new_AntEnv()
    return env

def make_walker(env='walker_2d'):
    assert env=='walker_2d'
    env = new_Walker2dEnv()
    return env

envs = {'Swimmer':make_swimmer, 'LunarLander-v2': make_lunar_env, 'CartPole-v0':make_cart_env,
        'half_cheetah': make_half_cheetah, 'Ant': make_ant, 'walker_2d': make_walker}

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
    ########## file related
    print(args.mass, args.goal)
    if args.mass == 1.0 and args.goal == 5.0:
      resdir = os.path.join(args.resdir, 'multimodalgoal',"")
    elif args.mass == 1.0 and args.goal == 10.0:
      resdir = os.path.join(args.resdir, 'uniformgoal',"")
    elif args.mass == 5.0:
      resdir = os.path.join(args.resdir, 'multimodal',"")
    elif args.mass == 10.0:
      resdir = os.path.join(args.resdir, 'uniform',"")
    else:
      resdir = os.path.join(args.resdir, 'simple',"")
 
    
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
    print(resdir)
    if not os.path.exists(resdir):
        
        os.makedirs(resdir)
    meta_rew_file = open(resdir + "EPIC_" + filename + ".txt", "w")

    # env = gym.make(env_name)
    envfunc = envs[env_name]
    env = envfunc(env_name)

    if learner == "vpg":
        print("-----initialize meta policy-------")
        actor_policy = GaussianVPG(env.observation_space, env.action_space,
                                  hidden_sizes=hidden_sizes, activation=activation, alpha=alpha,
                                  beta=beta, action_std=action_std, gamma=gamma, device=device,
                                  lam=lam, lam_decay=lam_decay)
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
        env = envfunc(env_name)
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
                    meta_rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                        sample, episode, np.round(np.sum(rewards), decimals = 3)))
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
                    # meta_rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                    #     sample, episode, np.round(np.sum(rewards), decimals=3)))
                    break

        actor_policy.update_mu_theta_for_default(meta_memory, meta_update_every, H=1*(1-gamma**max_steps)/(1-gamma))
        meta_memory.clear_memory()

        if (sample+1) % meta_update_every == 0:
            actor_policy.update_default_and_prior_policy()

        env.close()

    # rew_file.close()
    meta_rew_file.close()

            
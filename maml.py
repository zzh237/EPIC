import torch
import torch.nn as nn
import argparse
import gym
import mujoco_py
import numpy as np
import random
from gym.spaces import Box, Discrete
import setup
import os 
from algos.memory import Memory
from algos.agents.vpg import VPG
from algos.agents.ppo import PPO

from envs.new_cartpole import NewCartPoleEnv
from envs.new_lunar_lander import NewLunarLander
from envs.new_ant import AntDirection, AntForwardBackward
from envs.new_halfcheetah import HalfCheetahForwardBackward
from envs.new_humanoid import HumanoidDirection, HumanoidForwardBackward


import logging
from datetime import datetime
import copy

now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--run', type=int, default=-1)
# env settings Swimmer for majuto
parser.add_argument('--env', type=str, default="Swimmer")
parser.add_argument('--samples', type=int, default=2000) # need to tune
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--goal', type=float, default=0.0)
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
parser.add_argument('--alpha', type=float, default=1e-4, help="lr of single task policy")
parser.add_argument('--beta', type=float, default=1e-4, help="lr of meta policy")

parser.add_argument('--update_every', type=int, default=300)
parser.add_argument('--meta_update_every', type=int, default=50)  # need to tune
parser.add_argument('--hiddens', nargs='+', type=int)
# parser.add_argument('--grad_clip_norm', type=float, default=7.0)
# parser.add_argument('--iter_per_round', type=int, default=5)


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
    elif args.goal == 100:
      goalcart=np.random.uniform(-1,1)
      masscart=np.random.choice(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), p=[0.15,0.18,0.34,0.18,0.15])
      masspole=np.random.choice(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), p=[0.34,0.18, 0.18, 0.15, 0.15])
      length=np.random.choice(np.array([0.3, 0.4, 0.5, 0.6, 0.7]), p=[0.15,0.18,0.34,0.18,0.15])
      print("seed{}".format(seed))
      masscart = 0.1 * seed* np.random.randn() + masscart
      masspole = 0.01 * seed*np.random.rand() + masspole
      length = 0.01*seed*np.random.rand() + length
      env = NewCartPoleEnv(masscart=masscart,
                         masspole=masspole,
                         length=length,
                         goal=goalcart)
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

def make_car_env(seed, env="MountainCarContinuous-v0"):
    # need to tune
    env = gym.make("MountainCarContinuous-v0")
    return env

def make_half_cheetah(env='half_cheetah'):
    # assert env == 'half_cheetah', "env_name should be half_cheetah."
    env = HalfCheetahForwardBackward()
    return env

def make_antdirection(env):
    # assert env=='Ant'
    env = AntDirection()
    return env

def make_antforwardbackward(env):
    # assert env=='Ant'
    env = AntForwardBackward()
    return env

def make_humanoiddirection(env):
    env = HumanoidDirection()
    return env

def make_humanoidforwardbackward(env):
    env = HumanoidForwardBackward()
    return env



envs = {'LunarLander-v2': make_lunar_env,
        'CartPole-v0':make_cart_env,
        'AntDirection': make_antdirection,
        'AntForwardBackward': make_antforwardbackward,
        'HalfcheetahForwardBackward': make_half_cheetah,
        'HumanoidDirection': make_humanoiddirection,
        'HumanoidForwardBackward': make_humanoidforwardbackward,
        }

if __name__ == '__main__':
    ############## Hyperparameters ##############
    env_name = args.env  
    samples = args.samples
    max_episodes = args.episodes  # max training episodes
    max_steps = args.steps  # max timesteps in one episode
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
    # grad_clip_norm = args.grad_clip_norm
    # iter_per_round = args.iter_per_round
    ############ For All #########################
    gamma = 0.99  # discount factor
    render = False
    save_every = 100
    if args.hiddens:
        hidden_sizes = tuple(args.hiddens) # need to tune
    else:
        hidden_sizes = (256, 256)
    activation = nn.Tanh  # need to tune

    torch.cuda.empty_cache()
    ########## file related
    if args.mass == 1.0 and args.goal == 5.0:
      resdir = os.path.join(args.resdir, 'multimodalgoal',"")
    elif args.mass == 1.0 and args.goal == 10.0:
      resdir = os.path.join(args.resdir, 'uniformgoal',"")
    elif args.mass == 5.0:
      resdir = os.path.join(args.resdir, 'multimodal',"")
    elif args.mass == 10.0:
      resdir = os.path.join(args.resdir, 'uniform',"")
    elif args.mass == 100:
      resdir = os.path.join(args.resdir, 'dynamic',"")
    else:
      resdir = os.path.join(args.resdir, 'simple',"")
    filename = env_name + "_" + learner + "_s" + str(samples) + "_n" + str(max_episodes) \
        + "_every" + str(meta_update_every) \
        + "_size" + str(hidden_sizes[0]) \
            + "_goal" + str(args.goal)\
            + "_steps" + str(max_steps) \
            + "_mass" + str(args.mass)

    if args.run >= 0:
        filename += "_run" + str(args.run)

    
    if not os.path.exists(resdir):
        os.makedirs(resdir) 
    rew_file = open(resdir + "maml_" + filename + ".txt", "w")
    #meta_rew_file = open(args.resdir + "maml_" + "meta_" + filename + ".txt", "w")

    #env = make_env(env_name, args.seed)
    envfunc = envs[env_name]
    env = envfunc(0,env_name)
    if learner == "vpg":
        actor_policy = VPG(env.observation_space, env.action_space, hidden_sizes=hidden_sizes,
                           activation=activation, gamma=gamma, device=device, alpha=alpha,
                           beta=beta)
    if learner == "ppo":
        actor_policy = PPO(env.observation_space, env.action_space, hidden_sizes=hidden_sizes,
                           activation=activation, gamma=gamma, device=device, learning_rate=lr,
                           grad_clip_norm=grad_clip_norm, iter_per_round=iter_per_round)


    for sample in range(samples):
        meta_memory = Memory()
        print("#### MAML environment {} sample {}".format(env_name, sample))
        # env = make_env(env_name, sample)
        env = envfunc(sample,env_name)

        memory = Memory()

        start_episode = 0
        timestep = 0
        # for each task, we need to initialize policy_m
        actor_policy.initialize_policy_m()
        for episode in range(start_episode, max_episodes):
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
                memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
                state = new_state

                if done or steps == max_steps - 1:
                    rew_file.write("sample: {}, episode: {}, total reward: {}\n".format(
                        sample, episode, np.round(np.sum(rewards), decimals=3)))
                    break
        # obtain policy_m and apply gradient descent
        actor_policy.update_policy_m(memory)
        if learner == 'ppo': actor_policy.equalize_policies()
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

        actor_policy.update_policy(meta_memory)
        meta_memory.clear_memory()
        if (sample+1) % meta_update_every == 0:
            actor_policy.policy.load_state_dict(copy.deepcopy(actor_policy.new_policy.state_dict()))

        env.close()

    rew_file.close()
    # meta_rew_file.close()




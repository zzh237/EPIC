import argparse
import logging

# from datetime import datetime
## this is version 2.0
import os

# this needs to be set before any import of mujoco
os.environ["CC"] = "x86_64-conda-linux-gnu-gcc"

from functools import partial
from typing import Callable, Dict

import gym

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete

import wandb
from algos.agents.gaussian_ppo import GaussianPPO
from algos.agents.gaussian_vpg_mc import GaussianVPGMC
from algos.gaussian_sac import EpicSAC, FlattenStochasticMlp, KlRegularizationSettings, StochasticMlp
from algos.memory import Memory
from envs.new_ant import AntDirection, AntForwardBackward
from envs.new_cartpole import NewCartPoleEnv
from envs.new_halfcheetah import HalfCheetahForwardBackward
from envs.new_humanoid import HumanoidDirection, HumanoidForwardBackward
from envs.new_lunar_lander import NewLunarLander
from envs.new_swimmer import new_Swimmer
from algos.agents.sac_basic import VanillaSAC
import rlkit.torch.pytorch_util

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"


# use the conda-provided gcc12 to build cython stubs

# now = datetime.now()
# current_time = now.strftime("%m-%d %H:%M:%S")

parser = argparse.ArgumentParser()
# change cpu to cuda if running on server
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--run", type=int, default=0)
# env settings
# Swimmer for majuco environment
parser.add_argument(
    "--env",
    type=str,
    default="CartPole-v0",
    help=[
        "Swimmer",
        "LunarLander-v2",
        "CartPole-v0",
        "half_cheetah",
        "Ant",
    ],
)
parser.add_argument("--samples", type=int, default=2000)  # need to tune
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--goal", type=float, default=0.5)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--mass", type=float, default=1.0)
parser.add_argument("--action_std", type=float, default=0.5)
parser.add_argument("--m", type=int, default=10)
parser.add_argument("--c1", type=float, default=1.6)
# meta settings
parser.add_argument("--meta", dest="meta", action="store_true")
parser.add_argument("--no-meta", dest="meta", action="store_false")
parser.set_defaults(meta=True)
parser.add_argument("--meta-episodes", type=int, default=10)  # need to tune
parser.add_argument("--coeff", type=float, default=0.5)  # need to tune
parser.add_argument("--tau", type=float, default=0.5)  # need to tune

# learner settings
parser.add_argument("--learner", type=str, default="vpg", help="vpg, ppo, sac")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--beta", type=float, default=1e-4)
parser.add_argument("--update_every", type=int, default=300)
parser.add_argument("--meta_update_every", type=int, default=50)  # need to tune
parser.add_argument("--hiddens", nargs="+", type=int)
parser.add_argument("--lam", type=float, default=0.9)
parser.add_argument("--lam_decay", type=float, default=0.95)
## learner - sac
parser.add_argument("--replay-capacity", type=int, default=10_000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--q-kl-reg", action="store_true")
parser.add_argument("--v-kl-reg", action="store_true")
parser.add_argument("--policy-kl-reg", action="store_true")

## learner - vsac
parser.add_argument("--use-automatic-entropy-tuning", action="store_true")

# file settings
parser.add_argument("--logdir", type=str, default="logs/")
parser.add_argument("--resdir", type=str, default="results/")
parser.add_argument("--moddir", type=str, default="models/")
parser.add_argument("--loadfile", type=str, default="")

parser.add_argument("--render", action="store_true")

args = parser.parse_args()


def get_log(file_name):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(file_name, mode="a")
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def make_cart_env(seed, env="CartPole-v0"):
    assert env == "CartPole-v0", "env_name should be CartPole-v0."
    if args.mass == 5:
        masscart = np.random.choice(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        masspole = np.random.choice(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), p=[0.34, 0.18, 0.18, 0.15, 0.15])
        length = np.random.choice(np.array([0.3, 0.4, 0.5, 0.6, 0.7]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        masscart = 0.1 * np.random.randn() + masscart
        masspole = 0.01 * np.random.rand() + masspole
        length = 0.01 * np.random.rand() + length
        env = NewCartPoleEnv(masscart=masscart, masspole=masspole, length=length)
    elif args.mass == 10:
        masscart = np.random.uniform(1, 5)
        masspole = np.random.uniform(0.1, 0.5)
        length = np.random.uniform(0.3, 0.7)
        env = NewCartPoleEnv(masscart=masscart, masspole=masspole, length=length)
    elif args.goal == 5:
        goalcart = np.random.choice(np.array([-0.99, -0.5, 0, 0.5, 0.99]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        goalcart = 0.1 * np.random.randn() + goalcart
        env = NewCartPoleEnv(goal=goalcart)
    elif args.goal == 10:
        goalcart = np.random.uniform(-1, 1)
        env = NewCartPoleEnv(goal=goalcart)
    elif args.goal == 100:
        goalcart = np.random.uniform(-1, 1)
        masscart = np.random.choice(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        masspole = np.random.choice(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), p=[0.34, 0.18, 0.18, 0.15, 0.15])
        length = np.random.choice(np.array([0.3, 0.4, 0.5, 0.6, 0.7]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        print("seed{}".format(seed))
        masscart = 0.1 * seed * np.random.randn() + masscart
        masspole = 0.01 * seed * np.random.rand() + masspole
        length = 0.01 * seed * np.random.rand() + length
        env = NewCartPoleEnv(masscart=masscart, masspole=masspole, length=length, goal=goalcart)
    else:
        env = NewCartPoleEnv()
    return env


def make_lunar_env(seed, env="LunarLander-v2", continuous=False):
    if args.mass == 5:
        main_engine_power = np.random.choice(np.array([11.0, 12.0, 13.0, 14.0, 15.0]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        side_engine_power = np.random.choice(np.array([0.45, 0.55, 0.65, 0.75, 0.85]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        main_engine_power = main_engine_power + 0.1 * np.random.randn()
        side_engine_power = side_engine_power + 0.01 * np.random.randn()
        env = NewLunarLander(
            main_engine_power=main_engine_power, side_engine_power=side_engine_power, continuous=continuous
        )
    elif args.mass == 10:
        main_engine_power = np.random.uniform(3, 20)
        side_engine_power = np.random.uniform(0.15, 0.95)
        env = NewLunarLander(
            main_engine_power=main_engine_power, side_engine_power=side_engine_power, continuous=continuous
        )
    elif args.goal == 5:
        goal = np.random.choice(np.array([-0.99, -0.5, 0, 0.5, 0.99]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        goal = 0.1 * np.random.randn() + goal
        env = NewLunarLander(goal=goal, continuous=continuous)
    elif args.goal == 10:
        goal = np.random.uniform(-1, 1)
        env = NewLunarLander(goal=goal, continuous=continuous)
    else:
        env = NewLunarLander(continuous=continuous)
    # check_env(env, warn=True)
    return env


def make_car_env(env="MountainCarContinuous-v0"):
    # need to tune
    env = gym.make("MountainCarContinuous-v0")
    return env


def make_swimmer_env(env):
    goal = np.random.uniform(low=-0.5, high=0.5)
    env = new_Swimmer(goal)
    # env = SwimmerEnv()
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


def make_half_cheetah(env="half_cheetah"):
    # assert env == 'half_cheetah', "env_name should be half_cheetah."
    env = HalfCheetahForwardBackward()
    return env


def make_antdirection(seed, env):
    # assert env=='Ant'
    env = AntDirection()
    return env


def make_antforwardbackward(seed, env):
    # assert env=='Ant'
    env = AntForwardBackward()
    return env


def make_humanoiddirection(env):
    env = HumanoidDirection()
    return env


def make_humanoidforwardbackward(env):
    env = HumanoidForwardBackward()
    return env


def make_pendulum(seed, _):
    gravity = np.random.uniform(1.0, 20.0)

    env = gym.make("Pendulum-v1", g=gravity)
    env.seed(seed)
    return env

# def make_walker(env='walker_2d'):
#     assert env=='walker_2d'
#     env = new_Walker2dEnv()
#     return env

envs: Dict[str, Callable[..., gym.Env]] = {
    "LunarLander-v2": make_lunar_env,
    "LunarLander-v2-cont": partial(make_lunar_env, continuous=True),
    "CartPole-v0": make_cart_env,
    "AntDirection": make_antdirection,
    "AntForwardBackward": make_antforwardbackward,
    "HalfcheetahForwardBackward": make_half_cheetah,
    "HumanoidDirection": make_humanoiddirection,
    "HumanoidForwardBackward": make_humanoidforwardbackward,
    "Swimmer": make_swimmer_env,
    "Pendulum": make_pendulum
}

if __name__ == "__main__":
    ############## Hyperparameters ##############
    env_name = args.env  # "LunarLander-v2"
    # env_name = "LunarLander-v2"
    samples = args.samples
    max_episodes = args.episodes  # max training episodes
    max_steps = args.steps  # max timesteps in one episode
    meta_episodes = args.meta_episodes # episodes per environment
    learner = args.learner
    lr = args.lr

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
    gamma = 0.99  # discount factor
    render = args.render
    save_every = 100
    if args.hiddens:
        hidden_sizes = tuple(args.hiddens)  # need to tune
    else:
        hidden_sizes = (256, 256)
    activation = nn.Tanh  # need to tune
    rlkit.torch.pytorch_util.device = args.device

    use_model = False

    torch.cuda.empty_cache()

    wandb.init(project="epic-test")

    wandb.config.update(
        {
            "samples": samples,
            "max_episodes": max_episodes,
            "max_steps": max_steps,
            "meta_episodes": meta_episodes,
            "mass": args.mass,
            "goal": args.goal,
            "env_name": env_name,
            "learner": learner,
        }
    )

    ########## file related ####
    print(args.mass, args.goal)
    if args.mass == 1.0 and args.goal == 5.0:
        resdir = os.path.join(args.resdir, "multimodalgoal", "")
    elif args.mass == 1.0 and args.goal == 10.0:
        resdir = os.path.join(args.resdir, "uniformgoal", "")
    elif args.mass == 5.0:
        resdir = os.path.join(args.resdir, "multimodal", "")
    elif args.mass == 10.0:
        resdir = os.path.join(args.resdir, "uniform", "")
    elif args.mass == 100:
        resdir = os.path.join(args.resdir, "dynamic", "")
    else:
        resdir = os.path.join(args.resdir, "simple", "")

    filename = (
        env_name
        + "_"
        + learner
        + "_s"
        + str(samples)
        + "_n"
        + str(max_episodes)
        + "_every"
        + str(meta_update_every)
        + "_size"
        + str(hidden_sizes[0])
        + "_c"
        + str(coeff)
        + "_tau"
        + str(tau)
        + "_goal"
        + str(args.goal)
        + "_steps"
        + str(max_steps)
        + "_c1"
        + str(args.c1)
        + "_mc"
        + str(args.m)
        + "_lam"
        + str(args.lam)
    )

    if not use_meta:
        filename += "_nometa"

    if args.run >= 0:
        filename += "_run" + str(args.run)
    print(resdir)
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    meta_rew_file = open(resdir + "EPIC_" + filename + ".txt", "w")

    # env = gym.make(env_name)
    envfunc = envs[env_name]
    env: gym.Env = envfunc(1, env_name)
    m = args.m
    if learner == "vpg":
        print("-----initialize meta policy-------")
        actor_policy = GaussianVPGMC(
            env.observation_space,
            env.action_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
            alpha=alpha,
            beta=beta,
            action_std=action_std,
            gamma=gamma,
            device=device,
            lam=lam,
            lam_decay=lam_decay,
            m=m,
            c1=args.c1,
        )
    elif learner == "ppo":
        print("-----initialize meta policy-------")
        # here we could also use PPO, need to check difference between them
        actor_policy = GaussianPPO(
            env.observation_space,
            env.action_space,
            meta_update_every,
            hidden_sizes=hidden_sizes,
            activation=activation,
            gamma=gamma,
            device=device,
            learning_rate=lr,
            coeff=coeff,
            tau=tau,
            m=m,
        )
    elif learner.lower() == "sac":
        print("----initializing meta policy------")
        if isinstance(env, Discrete):
            raise ValueError("SAC does not support discrete action spaces")
    
        actor_policy = VanillaSAC(
            env=env,
            device=device,
            lr=lr,
            batch_size=args.batch_size,
            use_automatic_entropy_tuning=args.use_automatic_entropy_tuning,
            capacity=args.replay_capacity
        )
        

    elif learner.lower() == "epic-sac":
        print("-----initialize meta policy-------")
        if isinstance(env, Discrete):
            raise ValueError("SAC does not support discrete action spaces")
        # TODO don't hardcode hidden sizes
        actor_policy = EpicSAC(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            policy_hidden_sizes=(256, 256),
            policy_lr=lr,
            q_networks=[
                FlattenStochasticMlp(
                    input_size=env.observation_space.shape[0] + env.action_space.shape[0],
                    output_size=1,
                    hidden_dims=(256, 256),
                ).to(device)
                for _ in range(2)
            ],
            q_network_lr=lr,
            v_network=StochasticMlp(
                input_size=env.observation_space.shape[0], output_size=1, hidden_dims=(256, 256)
            ).to(device),
            v_network_lr=lr,
            device=device,
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            discount=args.discount,
            m=m,  # MC runs
            c1=args.c1,
            kl_settings=KlRegularizationSettings(
                q_network=args.q_kl_reg, policy=args.policy_kl_reg
            )
        )


    wandb.watch(actor_policy, log_freq=5)

    KL = 0
    for sample in range(samples):
        print("#### Learning environment {} sample {} ... workers ".format(env_name, sample), end="")
        ########## creating environment

        env = envfunc(sample, env_name)

        ########## sample a meta learner
        actor_policy.initialize_policy_m()  # initial policy theta

        mc_rewards = np.array([])
        start_episode = 0

        meta_memories = {}

        for j in range(m):
            print(f"{j} ", end="", flush=True)
            meta_memory = Memory()
            epi_reward = 0
            for episode in range(start_episode, meta_episodes):  # rollout multiple episodes on this env
                state = env.reset()
                rewards = []
                for step in range(max_steps):
                    if render and sample % 10 == 0:
                        env.render()
                    state_tensor, action_tensor, log_prob_tensor = actor_policy.act_policy_m(state, j)

                    if isinstance(env.action_space, Discrete):
                        action = action_tensor.item()
                    else:
                        action = action_tensor.cpu().data.numpy().flatten()
                    new_state, reward, done, _ = env.step(action)
                    rewards.append(reward)
                    meta_memory.add(state_tensor, action_tensor, log_prob_tensor, reward, done)
                    if isinstance(actor_policy, EpicSAC):
                        # particular MC actor
                        actor_policy.mc_actors[j].replay_buffer.push(state, action, reward, new_state, done)
                        mc_metrics = actor_policy.update_mu_theta_for_default(meta_memories, meta_update_every, H = 1* (1 - gamma ** max_steps) / (1 - gamma))
                    elif isinstance(actor_policy, VanillaSAC):
                        actor_policy.replay_buffer.push(state, action, reward, new_state, done)
                        mc_metrics = actor_policy.update_mu_theta_for_default(meta_memories, None, None)
                    state = new_state

                    if done or step == max_steps - 1:
                        epi_reward += np.sum(rewards)
                        break
            epi_reward = epi_reward / meta_episodes
            meta_memories[j] = meta_memory
            mc_rewards = np.append(mc_rewards, epi_reward)

        meta_rew_file.write(
            "sample: {}, mc_sample: {}, mean reward: {}, std reward: {}, kl: {}\n".format(
                sample,
                m,
                np.round(np.mean(mc_rewards), decimals=3),
                np.round(np.std(mc_rewards), decimals=3),
                np.round(KL, decimals=3),
            )
        )

        print(f"env mean reward {np.mean(mc_rewards):,.3f}")
        
        
        actor_policy.update_mu_theta_for_default(
            meta_memories, meta_update_every, H=1 * (1 - gamma**max_steps) / (1 - gamma)
        )

        if (sample + 1) % meta_update_every == 0:
            actor_policy.update_default_and_prior_policy()

        wandb.log({"env_number": sample, "reward": {"mean": np.mean(mc_rewards), "std": np.std(mc_rewards)}})

        env.close()

    # rew_file.close()
    meta_rew_file.close()

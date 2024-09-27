"""
Rewrite of epic_mc to simplify some logic.
"""

from __future__ import annotations

import random
from argparse import ArgumentParser
from functools import partial
from typing import Callable

import gym
import numpy as np
import polars as pl
import rlkit.torch.pytorch_util
import torch
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW

import wandb
from algos.agents.gaussian_ppo_2 import GaussianPPO2
from algos.agents.gaussian_vpg_2 import GaussianVPGMC2
from algos.agents.sac_basic import VanillaSACv2
from algos.gaussian_sac2 import EpicSAC2
from algos.types import EPICModel
from envs import make_pendulum
from envs.jellybean import make_jbw
from envs.swimmer_epic_2 import make_swimmer
from envs.lunar_epic_2 import make_lunar_env
# import gtimer as gt


def parse_args():
    parser = ArgumentParser("epic_2")
    parser.add_argument("--wandb-project", type=str, default="epic-test")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)

    # common parameters
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--m", help="num mc workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)

    # "metalearning"
    parser.add_argument("--meta-episodes", type=int, default=10)
    parser.add_argument("--meta-update-every", type=int, default=5)

    # EPIC regularizer params
    parser.add_argument("--enable-epic-regularization", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--c", type=float, default=1.5)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--prior-lambda", type=float, default=0.9)
    parser.add_argument("--prior-lambda-decay", type=float, default=0.999)

    # SAC - style algorithms
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr-qf", type=float)
    parser.add_argument("--lr-policy", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--qf-target-update-period", type=int)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--sac-steps", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1)
    parser.add_argument("--use-automatic-entropy-tuning", action="store_true")
    parser.add_argument("--replay-capacity", type=int, default=1_000)

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=10032)

    ret = parser.parse_args()

    rlkit.torch.pytorch_util.device = torch.device(ret.device)

    return ret


class EpicTrainer:
    def __init__(
        self,
        model: EPICModel,
        env_maker: Callable[[int], gym.Env],
        meta_episodes: int,
        num_episodes: int,
        max_steps: int,
        render: bool,
        meta_update_every: int,
    ):
        self.model = model
        self.env_maker = env_maker
        self.meta_episodes = meta_episodes  # number of environments per lifetime
        self.num_episodes = num_episodes  # number of episodes per environment
        self.max_steps = max_steps  # max steps per episode
        self.render = render
        self.meta_update_every = meta_update_every  # update the prior every this many meta-episodes


    def train_and_evaluate(self):
        # [(meta-episode / life episode) -> episode -> mcworker -> step]
        rewards = pl.DataFrame(
            schema={"meta_episode": int, "episode": int, "mc_worker": int, "step": int, "reward": pl.Float64}
        )
        for meta_episode in range(self.meta_episodes):
            env = self.env_maker(meta_episode)
            print(f"meta-episode: {meta_episode}, episodes:", end="")
            self.model.pre_meta_episode()
            for episode_idx in (range(self.num_episodes)):
                for m_idx in range(self.model.m):
                    state = env.reset()
                    for step in range(self.max_steps):
                        if self.render:
                            env.render()

                        action_out = self.model.act_m(m_idx, state)
                        action = action_out["action"].detach().cpu().numpy()
                        new_state, reward, done, _ = env.step(action)
                        rewards.extend(
                            pl.DataFrame(
                                {
                                    "meta_episode": meta_episode,
                                    "episode": episode_idx,
                                    "mc_worker": m_idx,
                                    "step": step,
                                    "reward": reward,
                                }
                            )
                        )

                        self.model.per_step_m(m_idx, meta_episode, step, action_out, reward, new_state, done)

                        state = new_state
                        if done or step == (self.max_steps - 1):
                            break
                self.model.post_episode()
                print(f" {episode_idx}", flush=True, end="")
            
            # TODO let policy gradient models hook into the end of the meta-episode so they can consume all
            # the trajectories without triggering a default / prior update

            # average reward over all mc workers for this meta-episode
            meta_episode_reward = (
                rewards.filter(pl.col("meta_episode") == meta_episode)
                .group_by("meta_episode")
                .agg(pl.col("reward").mean())
            ).item(row=0, column=1)
            print(f", reward: {meta_episode_reward}")
            wandb.log({"meta_episode_reward": meta_episode_reward, "meta_episode": meta_episode})

            if (meta_episode + 1) % self.meta_update_every == 0:
                self.model.update_default()
                self.model.update_prior()
            # gt.stamp("meta-episode")


def make_model(args, env) -> EPICModel:
    if args.optimizer.lower() == "adam":
        optimizer = Adam
    elif args.optimizer == "sgd":
        optimizer = SGD
    elif args.optimizer == "adamw":
        optimizer = AdamW
    else:
        raise ValueError(f"Unrecognized optimizer {args.optimizer}")

    if args.model == "vsac":
        mdl = VanillaSACv2(
            m=args.m,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            use_automatic_entropy_tuning=args.use_automatic_entropy_tuning,
            replay_capacity=args.replay_capacity,
            env=env,
        )
        wandb.watch(mdl)
        return mdl
    elif args.model == "epic-sac":
        mdl = EpicSAC2(
            m=args.m,
            env=env,
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            sac_steps=args.sac_steps,
            optimizer_class=optimizer,
            policy_lr=args.lr_policy,
            qf_lr=args.lr_qf,
            tau=args.tau,
            qf_target_update_period=args.qf_target_update_period,
            reward_scale=args.reward_scale,
            discount=args.discount,
            device=args.device,
            prior_update_every=args.meta_update_every,
            max_steps=args.max_steps,
            gamma=args.gamma,
            c = args.c,
            delta=args.delta,
            enable_epic_regularization=args.enable_epic_regularization,
            prior_lambda=args.prior_lambda,
            prior_lambda_decay=args.prior_lambda_decay
        )
        wandb.watch(mdl)
        return mdl
    elif args.model == "gaussian-ppo":
        mdl = GaussianPPO2(
            state_space=env.observation_space,
            action_space=env.action_space,
            meta_update_every=args.meta_update_every,
            device=args.device,
            learning_rate=args.lr
        )
        wandb.watch(mdl)
        return mdl
    elif args.model == "gaussian-vpg":
        mdl = GaussianVPGMC2(
            m=args.m,
            env=env,
            hidden_sizes=(256, 256),
            device=args.device,
            action_std=0.5,
            prior_update_every=args.meta_update_every,
            gamma=args.gamma,
            max_steps=args.max_steps,
            c=args.c,
            delta=args.delta,
            tau=args.tau,
            lr=args.lr,
            c1=1.6,
            lam=args.prior_lambda,
            lam_decay=args.prior_lambda_decay,
            enable_epic_regularization=args.enable_epic_regularization,
            optimizer=optimizer
        )
        wandb.watch(mdl)
        return mdl
    else:
        raise ValueError(f"Unrecognized model type {args.model}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ENV_MAKERS: dict[str, Callable[[int], gym.Env]] = {
    "pendulum": partial(make_pendulum, toy=False),
    "pendulum-toy": partial(make_pendulum, toy=True),
    "jbw": partial(make_jbw, render=args.render, period=args.max_steps, proper_reset=True),
    "swimmer": make_swimmer,
    "lunar-cont": partial(make_lunar_env, continuous=True)
}
    wandb.init(project=args.wandb_project)
    # env instantiation is just for space dimension
    model = make_model(args, env=ENV_MAKERS[args.env](0))

    # gt.start()

    trainer = EpicTrainer(
        model,
        env_maker=ENV_MAKERS[args.env],
        meta_episodes=args.meta_episodes,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=args.render,
        meta_update_every=args.meta_update_every,
    )

    trainer.train_and_evaluate()
    # gt.stamp("train-and-evaluate")

    # print(gt.report())


if __name__ == "__main__":
    main()

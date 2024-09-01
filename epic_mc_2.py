"""
Rewrite of epic_mc to simplify some logic.
"""

from __future__ import annotations

import torch
from typing import Callable


import gym
import numpy as np
import random
from argparse import ArgumentParser
from functools import partial

import wandb
from algos.agents.sac_basic import VanillaSACv2
from algos.gaussian_sac import EpicSAC2
from algos.types import EPICModel
from envs import make_pendulum
import polars as pl



def parse_args():
    parser = ArgumentParser("epic_2")
    parser.add_argument("--wandb-project", type=str, default="epic-test")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--env", type=str)

    # common parameters
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--m", help="num mc workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)

    # metalearning
    parser.add_argument("--meta-episodes", type=int, default=10)
    parser.add_argument("--meta-update-every", type=int, default=5)

    # SAC - style algorithms
    parser.add_argument("--use-automatic-entropy-tuning", action="store_true")
    parser.add_argument("--replay-capacity", type=int, default=1_000)

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=10032)

    return parser.parse_args()


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
            print(f"meta-episode: {meta_episode}, episodes:", end="")
            env = self.env_maker(meta_episode)

            for episode_idx in range(self.num_episodes):
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

                        self.model.per_step_m(m_idx, state, action, reward.item(), new_state, done, episode_idx, step)

                        state = new_state

                        if done or step == (self.max_steps - 1):
                            break
                self.model.post_episode()
                print(f" {episode_idx}", flush=True, end="")
            # average reward over all mc workers for this meta-episode
            meta_episode_reward = (
                rewards.filter(pl.col("meta_episode") == meta_episode).group_by("meta_episode").agg(pl.col("reward").mean())
            ).item(row=0, column=1)
            print(f", reward: {meta_episode_reward}")
            wandb.log({"meta_episode_reward": meta_episode_reward, "meta_episode": meta_episode})

            if (meta_episode + 1) % self.meta_update_every == 0:
                self.model.update_default()
                self.model.update_prior()


def make_model(args, env) -> EPICModel:
    if args.model == "vsac":
        return VanillaSACv2(
            m=args.m,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            use_automatic_entropy_tuning=args.use_automatic_entropy_tuning,
            replay_capacity=args.replay_capacity,
            env=env,
        )
    elif args.model == "epic-sac":
        return EpicSAC2(

        )
    else:
        raise ValueError(f"Unrecognized model type {args.model}")


ENV_MAKERS: dict[str, Callable[[int], gym.Env]] = {"pendulum": make_pendulum,
                                                   "pendulum-toy": partial(make_pendulum, toy=True)}


def main():
    import rlkit.torch.pytorch_util
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rlkit.torch.pytorch_util.set_gpu_mode(True)

    wandb.init(project=args.wandb_project)
    # just for space dimensions
    model = make_model(args, env=ENV_MAKERS[args.env](0))

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


if __name__ == "__main__":
    main()

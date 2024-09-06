# Pendulum - toy
Note that the full-episode reward on a solved pendulum is like ~-1.8

converging the vanilla SAC on the simple pendulum environment

```shell
python ./epic_mc_2.py --model=vsac --device=cpu --env=pendulum-toy --max-steps=200
```

toy - converging stochastic SAC on the simple pendulum environment
(kinda unstable, but converges)

```shell
python ./epic_mc_2.py \
--model=epic-sac \
--device=cpu \
--env=pendulum-toy \
--max-steps=200 \
--lr-qf=5e-3 \
--lr-policy=5e-3 \
--qf-target-update-period=1 \
--replay-capacity=1000 \
--batch-size=32 \
--tau=1e-2 --render
```
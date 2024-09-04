toy - converging the vanilla SAC on the simple pendulum environment

```shell
python ./epic_mc_2.py --model=vsac --device=cpu --env=pendulum-toy --max-steps=200
```

toy - converging stochastic SAC on the simple pendulum environment
(this doesn't converge yet...)

```shell
python ./epic_mc_2.py \
--model=epic-sac \
--device=cpu \
--env=pendulum-toy \
--max-steps=200 \
--lr-qf=1e-3 \
--lr-policy=1e-3 \
--qf-target-update-period=1 \
--tau=1e-2
```
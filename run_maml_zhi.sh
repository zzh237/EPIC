#! /bin/bash
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/home/z/zzh237/.mujoco/mujoco210/bin
device="cuda"
if [ $device != "cpu" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
fi
source ~/.bashrc

RUNS=1
for env_name in  "CartPole-v0"
do
for n in 25 
do 
for step in 100
do
for mass in 100
do
for ((i=0;i<${RUNS};i++));
do
    python maml.py --run ${i} --env "${env_name}" --device "${device}" --samples 1000 --meta_update_every $n --steps $step --goal 100 --mass $mass --resdir "results_maml/${env_name}/"
done
done
done
done
done
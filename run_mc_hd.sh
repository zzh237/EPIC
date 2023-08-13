#! /bin/bash
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/home/z/zzh237/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
source ~/.bashrc

RUNS=1
device="cuda"
for n in 5  
do
for step in 2000
do
for m in 10 
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "HumanoidDirection" --device "${device}" --meta_update_every $n --steps $step --m $m --resdir "results/montecarlo/step${step}_8/"
done
done 
done
done
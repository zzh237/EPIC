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
for env_name in  "HalfcheetahForwardBackward"
do
for n in 5 
do
for step in 1000
do
for m in 10 
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "${env_name}" --device "${device}" --samples 1000 --meta_update_every $n --samples 1000 --steps $step --m $m --resdir "results/montecarlo/${env_name}/"
done
done 
done
done
done
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
for env_name in  "AntForwardBackward"
do
for n in 5 10 25 50 
do
for step in 1000
do
for m in 1 
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "${env_name}" --device "${device}" --meta_update_every $n --steps $step --m $m --resdir "results/montecarlo/${env_name}/"
done
done 
done
done
done
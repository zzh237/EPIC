#! /bin/bash
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/home/z/zzh237/.mujoco/mujoco210/bin
device="cpu"
if [ $device != "cpu" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
fi
source ~/.bashrc

RUNS=3
for env_name in  "AntForwardBackward"
do
for n in 25 
do 
for step in 1000
do
for mass in 1.0
do
for ((i=0;i<${RUNS};i++));
do
    python maml.py --run ${i} --env "${env_name}" --device "${device}" --meta_update_every $n --steps $step --mass $mass --resdir "results_maml/${env_name}/"
done
done
done
done
done
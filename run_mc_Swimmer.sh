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

RUNS=1
for n in 25 
do
for step in 100
do
for m in 10 
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "Swimmer" --device "${device}" --meta_update_every $n --steps $step --mass 5 --m $m --goal 10.0 --resdir "results/montecarlo/step${step}_9/"
done
done 
done
done
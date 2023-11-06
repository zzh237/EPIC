#! /bin/bash 


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/home/z/zzh237/.mujoco/mujoco210/bin
device="cuda"
if [ $device != "cpu" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
fi
source ~/.bashrc


RUNS=2
for n in 10 
do
for step in 1000
do
for m in 10
do
for ((i=1;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "AntDirection" --samples 1000 --c1 1.9 --device "${device}" --meta_update_every $n --steps $step --m $m --resdir "results/montecarlo/${env_name}/"
done
done 
done
done
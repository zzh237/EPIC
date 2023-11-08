#! /bin/bash  

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/home/z/zzh237/.mujoco/mujoco210/bin
device="cuda"
if [ $device != "cpu" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
fi
source ~/.bashrc
RUNS=1
env="CartPole-v0"
for n in 25 
do
for step in 100
do
for m in 10 
do
for lam in 0.9
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --lam $lam --env ${env} --meta_update_every $n --steps $step --mass 100 --m $m --goal 100 --c1 1.9 --samples 1000 --resdir "results/montecarlo/${env}/" --device "$device"
done
done 
done 
done
done

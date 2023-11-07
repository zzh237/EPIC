#! /bin/bash  
RUNS=1
device="cuda"
env="CartPole-v0"
for n in 25 
do
for step in 100
do
for m in 15 
do
for lam in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --lam $lam --env ${env} --meta_update_every $n --steps $step --mass 5 --m $m --goal 10.0 --resdir "results/montecarlo/${env}/" --device "$device"
done
done 
done 
done
done

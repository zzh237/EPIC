#! /bin/bash  
RUNS=1
for n in 25
do
for step in 300
do
for m in 100
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "CartPole-v0" --meta_update_every $n --steps $step --mass 5 --m $m --goal 10.0 --resdir "results/montecarlo/new1/"
done
done 
done
done
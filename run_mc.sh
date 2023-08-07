#! /bin/bash  
RUNS=1
for n in 25
do
for step in 100
do
for m in 10
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "CartPole-v0" --meta_update_every $n --steps $step --mass 10 --m $m --goal 10.0 --resdir "results/montecarlo/step${step}/"
done
done 
done
done
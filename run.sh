#! /bin/bash  
RUNS=1
for n in 25
do
for step in 300
do
for m in 500
do
for ((i=0;i<${RUNS};i++));
do
    python epic.py --run ${i} --env "CartPole-v0" --meta_update_every $n --steps $step --mass 5 --m $m --goal 10.0 --resdir "results/montecarlo/"
done
done 
done
done
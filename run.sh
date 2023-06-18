#! /bin/bash  
RUNS=3
for n in 25
do
for step in 100 300
do
for e in 10
do
for ((i=0;i<${RUNS};i++));
do
    python epic.py --run ${i} --env "CartPole-v0" --meta_update_every $n --meta-episodes $e --steps $step --mass 5 --goal 10.0 --resdir "results/test/single_nokl"
done 
done
done
done
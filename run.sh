#! /bin/bash  
RUNS=2
for n in 25
do
for step in 100 300
do
for e in 10
do
for ((i=0;i<${RUNS};i++));
do
    python epic.py --run ${i} --env "CartPole-v0" --meta_update_every $n --meta-episodes $e --steps $step --mass 5 --goal 10.0 --resdir "results/test/nosingle_kl/default"
done 
done
done
done
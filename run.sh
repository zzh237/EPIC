#! /bin/bash  
RUNS=1
for n in 25
do
for step in 300
do
for mass in 1
do
for ((i=0;i<${RUNS};i++));
do
    python epic.py --run ${i} --env "Ant" --meta_update_every $n --steps $step --mass $mass --goal 10.0 --resdir "results/ant2/"
done
done 
done
done
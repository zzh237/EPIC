#! /bin/bash  
RUNS=1
for n in 5 10 25 50
do
for step in 1000
do
for m in 1 5 10 15
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "AntDirection" --meta_update_every $n --steps $step --mass 10 --m $m --goal 10.0 --resdir "results/montecarlo/step${step}/"
done
done 
done
done
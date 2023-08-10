#! /bin/bash  
RUNS=1
for n in 5 10 25 50
do
for step in 1000
do
for m in 10
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "HalfcheetahForwardBackward" --device "cuda:0" --meta_update_every $n --steps $step --m $m --resdir "results/montecarlo/step${step}/"
done
done 
done
done
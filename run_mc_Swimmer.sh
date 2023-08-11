#! /bin/bash  
RUNS=1
device="cuda"
for n in 25 
do
for step in 100
do
for m in 10 
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py --run ${i} --env "Swimmer" --meta_update_every $n --steps $step --mass 5 --m $m --goal 10.0 --resdir "results/montecarlo/step${step}_7/"
done
done 
done
done
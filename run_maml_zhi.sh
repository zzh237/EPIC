#! /bin/bash  
RUNS=2
for env_name in  "HumanoidDirection" "HumanoidForwardBackward"
do
for n in 5 10 25 50
do 
for step in 1000
do
for mass in 1.0
do
for ((i=0;i<${RUNS};i++));
do
    python maml.py --run ${i} --env $env_name --device "cuda:1" --meta_update_every $n --steps $step --mass $mass --goal 10.0 --resdir "results_maml/ant2/"
done
done
done
done
done
#! /bin/bash  
RUNS=19
for env_name in "AntForwardBackward" "HalfcheetahForwardBackward" "HumanoidDirection" "HumanoidForwardBackward"
do
for n in 5 10 25 50
do 
for step in 1000
do
for mass in 1.0
do
for ((i=18;i<${RUNS};i++));
do
    python maml.py --run ${i} --env $env_name --samples 1000 --device "cuda:1" --meta_update_every $n --steps $step --mass $mass --goal 10.0 --resdir "results_maml/ant2/"
done
done
done
done
done
#! /bin/bash  
RUNS=6
for env_name in "AntDirection" "AntForwardBackward" "HalfcheetahForwardBackward" "HumanoidDirection" "HumanoidForwardBackward"
do
for n in 5 10 25 50
do 
for step in 1000
do
for mass in 1.0
do
for ((i=1;i<${RUNS};i++));
do
    python maml.py --run ${i} --env $env_name --samples 1000 --device "cuda:0" --meta_update_every $n --steps $step --mass $mass --goal 10.0 --resdir "results_maml/ant2/"
done
done
done
done
done
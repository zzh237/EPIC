#! /bin/bash  
RUNS=1
for ((i=0;i<${RUNS};i++));
do
    python3 meta.py --run ${i} --env "CartPole-v0"
done
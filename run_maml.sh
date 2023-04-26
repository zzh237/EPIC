#! /bin/bash  
RUNS=10
for ((i=0;i<${RUNS};i++));
do
    python3 maml.py --run ${i} --env "Swimmer"
done
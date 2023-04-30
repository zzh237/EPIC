#! /bin/bash  
RUNS=1
for n in 1 5 10 25 
do 
for ((i=0;i<${RUNS};i++));
do
    python maml.py --run ${i} --env "CartPole-v0" --meta_update_every $n --steps 500
done
done 
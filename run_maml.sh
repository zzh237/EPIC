#! /bin/bash  
RUNS=1
for n in 10 25 50 100 200  
do 
for ((i=0;i<${RUNS};i++));
do
    python maml.py --run ${i} --env "CartPole-v0" --meta_update_every $n --steps 100
done
done 
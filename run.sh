#! /bin/bash  
RUNS=5
for n in 5 10 25 50
do 
for ((i=0;i<${RUNS};i++));
do
    python epic.py --run ${i} --env "half_cheetah" --meta_update_every $n --steps 100
done
done 
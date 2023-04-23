#! /bin/bash  
RUNS=10
for ((i=0;i<${RUNS};i++));
do
    python meta.py --run ${i} --no-meta
done
#!/usr/bin/env bash
RUNS=1
device="cpu"

for n in 25 
do
for step in 300
do
for m in 1
do
for ((i=0;i<${RUNS};i++));
do
    python epic_mc.py \
    --run ${i} \
    --env "jbw" \
    --device "${device}" \
    --meta_update_every $n \
    --steps $step \
    --mass 5 --m $m --goal 10.0 \
    --resdir "results/montecarlo/step${step}_9/"
done
done 
done
done
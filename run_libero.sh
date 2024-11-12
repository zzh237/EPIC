#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblogs/joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=20:00:00,h_data=8G,cuda=1,gpu,h_vmem=10G,RTX2080Ti
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 4

source /u/local/Modules/default/init/modules.sh

module load gcc/11.3.0
module load cuda/12.3

source activate_pixi.sh
time python ./scripts/libero_main.py
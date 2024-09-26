#!/bin/bash
#$ -cwd
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=5:00:00,h_data=16G
## Modify the parallel environment
## and the number of cores as needed
#$ -pe shared 4

# this script takes two arguments: the agent ID and the number of sweeps to execute

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

source /u/local/Modules/default/init/modules.sh
module load gcc/11.3.0

source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
WANDB_MODE=online pixi run wandb agent $1 --count $2
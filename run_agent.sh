#!/bin/bash
#$ -cwd
#$ -o qsub_log/joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
## needs an intel-gold processor for polars compatibility
#$ -l h_rt=5:00:00,h_data=16G,arch=intel-gold\*
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
# module load cuda/12.3

source ~/.bashrc
source ./activate_pixi.sh
# too hard to install the whole openGL sdk needed for mujoco nvidia support
# we can't render on these workers anyways, so disable GPU utilization
export MUJOCO_PY_FORCE_CPU=1
# adding /usr/lib/nvidia doesn't do anything on the cluster (load cuda above sets all of that), but 
# mujoco expects its presence anyways
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
WANDB_MODE=online wandb agent $1 --count $2
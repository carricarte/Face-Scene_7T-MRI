#!/bin/bash

#SBATCH --job-name=retinotopy
#SBATCH --mem-per-cpu=3GB
#SBATCH --cpus-per-task=20
#SBATCH --time=3-00:00:00
#SBATCH --qos=prio

sub=1

#Extract parameters
echo sub $sub
echo depth $depth
echo using $SLURM_CPUS_PER_TASK cpus

cd /home/mayaaj90/scripts/project-01-LaminarCrowdpRF/

### Start job
matlab -nosplash -noFigureWindows -r "pRFmapping_curta(${sub},${depth})" > pRFjob.out
echo set to run
### Output core and memory efficiency

seff $SLURM_JOBID

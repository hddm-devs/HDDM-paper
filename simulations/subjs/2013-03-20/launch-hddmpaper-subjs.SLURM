#!/bin/sh
#SBATCH -J hddm-subjs
#SBATCH --time=12:00:00
#SARRAY --range=1-24

# information that may be useful for debugging
echo Running hddm-subjs
echo "Starting job $SLURM_ARRAYID on $HOSTNAME"
echo Directory is `pwd`
echo Starting execution at `date`


module load python/2.7.3
cd ~/python/HDDM-paper
sleep $[4*$SLURM_ARRAYID]

echo "Launching job"
python ~/python/HDDM-paper/run_estimation.py -r --subjs

echo "Done!!!"
echo Finished execution at `date`


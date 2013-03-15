#!/bin/sh
#SBATCH -J collect-regress
#SBATCH --time=2:00:00
#SBATCH -n 1

# information that may be useful for debugging
echo Running collect-regress
echo "Starting job on $HOSTNAME"
echo Directory is `pwd`
echo Starting execution at `date`


module load python/2.7.3
cd ~/python/HDDM-paper
echo "Launching job"
python ~/python/HDDM-paper/run_estimation.py -r --regress --action collect

echo "Done!!!"
echo Finished execution at `date`


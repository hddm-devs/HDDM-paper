#!/bin/sh
#PBS -N hddm-paper
#PBS -r n
#PBS -l nodes=20:ppn=8
#PBS -l walltime=10:00:00

source hddm_venv2/bin/activate

cd HDDM-paper

echo "Launching controller"
ipcontroller --profile=mpi &
sleep 10

echo "Launching engines"
mpirun -np 160 --machinefile $PBS_NODEFILE -x PATH -x PYTHON_PATH -x VIRTUAL_ENV -wd ~/HDDM-paper ipengine --profile=mpi &
sleep 10

echo "Launching job"
python ~/HDDM-paper/run_estimation.py -r -a --profile mpi --parallel --all -st -sz -z

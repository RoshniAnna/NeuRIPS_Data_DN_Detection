#!/bin/bash
  
#SBATCH     --partition=gpu-preempt
#SBATCH     --nodes=1
#SBATCH     --output=location8500_EV_%A.txt
#SBATCH     --array=0  # for 3 data 
#SBATCH     --ntasks=1
#SBATCH     --time=2-23:00:00
#SBATCH     --mail-type=ALL
#SBATCH     --mail-user=mxu200014@utdallas.edu

data_type=("PVAttacks" "EVCSAttacks" "SensorAttacks")

prun python train_location_acc.py --data_type ${data_type[$SLURM_ARRAY_TASK_ID]}

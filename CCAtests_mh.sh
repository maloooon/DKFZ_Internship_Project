#!/bin/bash
#SBATCH --time=0-01:20:00 # maximum allocated time
#SBATCH --job-name=CCAtests_mh # name of the job
#SBATCH --partition=gpu-12h # which partition the job should be scheduled on
#SBATCH --output=./imgAE-%j.out
#SBATCH --error=./imgAE-%j.err
#SBATCH -w gpu[25-28]

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 ./imgAE.py 

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print
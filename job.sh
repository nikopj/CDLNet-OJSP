#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=3GB
#SBATCH --job-name=dncnnd
#SBATCH --mail-type=END
#SBATCH --mail-user=npj226@nyu.edu

module load cuda/11.1.74
source /scratch/npj226/.ptmri/bin/activate
cd /scratch/npj226/CDLNet-OJSP
python train.py args.json


#!/bin/bash

#SBATCH -p gpu 
#SBATCH -c 8
#SBATCH --gres=gpu:ampere:1
##SBATCH -p preempt
##SBATCH --qos preempt
##SBATCH --requeue
##SBATCH --open-mode=append
##SBATCH --cpus-per-gpu=8
##SBTACH --mem-per-cpu=8G
##SBATCH -C gmem80
##SBATCH -C '(turing|ampere) &(gmem48|gmem32)'
#SBATCH --job-name=MVIT2_DET
#SBATCH --output=train_mvit.out
#SBATCH --gres-flags=enforce-binding
##SBATCH --mail-type=BEGIN,END,FAIL     
##SBATCH --mail-user rajatmodi62@gmail.com

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
module load anaconda3
module load ffmpeg 

#module load cuda/9.0
#module load gcc/5
source activate slowfast
python train_mvit_detector_jhmdb.py



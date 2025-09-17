#!/bin/bash

#SBATCH --job-name=videollava_ek100_incontext

#SBATCH --output=sbatch_logs/videollava_ek100_incontext_%j.out

#SBATCH --error=sbatch_logs/videollava_ek100_incontext_%j.err

#SBATCH --nodes=1

#SBATCH --partition=gpu

#SBATCH --gres=gpu:v100:1

#SBATCH --mem=16G

#SBATCH --time=5-0:00:00 

#SBATCH --qos=batch-short

#SBATCH --mail-type=END

#SBATCH --mail-user=thao.le@deakin.edu.au

source activate videollava

python video-llava/video_llava_ek100_incontext.py
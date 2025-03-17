#!/bin/bash
#SBATCH --account=cseduimc030
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --gres=gpu:5
#SBATCH --time=48:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err

# Download the dataset
python data/download_data.py


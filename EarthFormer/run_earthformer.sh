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

conda create --name ef_sat2rad --file ef-sat2rad/Preprocess/ef_sat2rad.txt
conda activate ef_sat2rad

# Train the model
python ef-sat2rad/train_cuboid_sevir_invLinear.py

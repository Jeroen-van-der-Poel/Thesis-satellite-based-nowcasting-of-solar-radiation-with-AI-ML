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

# Create env
bash install_python_ubuntu.sh
bash create_environment.sh
source .venv/bin/activate

# Download the dataset
python data/download_data.py

# Train the model
python gptcast/train.py trainer=gpu experiment=vaeganvq_mwae.yaml 
# python gptcast/train.py trainer=gpu experiment=gptcast_16x16.yaml model.first_stage.ckpt_path=<path_to_vae_checkpoint>

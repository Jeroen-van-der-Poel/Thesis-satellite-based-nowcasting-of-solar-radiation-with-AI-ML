#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err

module load Miniconda3  
source ~/.bashrc

ENV_NAME=ef_env

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.9
    conda activate $ENV_NAME

    # Install dependencies
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install pytorch_lightning==2.5.1
    pip install torchmetrics==2.5.1
    pip install xarray netcdf4 opencv-python earthnet==0.3.9
    pip install -U -e . --no-build-isolation
else
    echo "Conda environment $ENV_NAME already exists."
    conda activate $ENV_NAME
fi

cd $HOME/projects/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer


# Train the model
python train.py --gpu 0 --cfg config/train.yml --ckpt_name last.ckpt --save ef_v1
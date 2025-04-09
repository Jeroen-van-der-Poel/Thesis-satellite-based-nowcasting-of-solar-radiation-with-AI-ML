# Satellite-based Nowcasting of Solar Radiation with AI/ML

This repository contains code and data pipelines for experimenting with state-of-the-art deep learning models for short-term solar radiation nowcasting, using geostationary satellite images. It includes implementations and training workflows for both **DGMR-SO** and **Earthformer** architectures.


## Data Preperation

## DGMR-SO 

DGMR-SO is an adapted version of the Deep Generative Model for Radar (DGMR) by Google DeepMind, tailored for satellite-based **Surface Solar Irradiance (SSI)** nowcasting. Instead of radar data, this implementation uses geostationary satellite imagery from the SEVIRI instrument aboard the Meteosat-11 satellite, normalized to represent clear-sky index for improved stability. The task is to predict the next 4 hours (16 future timesteps) of SSI based on the past 1 hour (4 timesteps) using rolling windows.

The original paper: https://www.sciencedirect.com/science/article/pii/S0038092X24005619#b0240

### Installation

!IMPORTANT make sure conda, cuda and cuDNN are avaialble/downloaded!

1. Clone the GitHub repository
2. Move into DGMR_SO folder
3. Create conda environment: "conda create -n dgmr_env python=3.9"
4. Activate conda environment: "conda activate dgmr_env"
5. Install the required packages:
     - "pip install tensorflow[and-cuda]"
     - "pip install matplotlib"
     - "pip install dm-sonnet"
     - "pip install pyyaml"

After these steps you should be able the execute the next steps.

### Training

## EarthFormer
Earthformer is a space-time transformer architecture designed for Earth system forecasting tasks, such as weather nowcasting and precipitation prediction. Unlike traditional CNN or ConvLSTM-based models, Earthformer leverages spatiotemporal attention mechanisms to effectively capture both short- and long-range dependencies across space and time. The transformer Utilizes axial attention and factorized self-attention to reduce complexity while preserving global context. In this project, Earthformer is adapted to nowcast Surface Solar Irradiance (SSI) using satellite-derived input data.

The original paper: https://www.amazon.science/publications/earthformer-exploring-space-time-transformers-for-earth-system-forecasting

### Installation

### Training

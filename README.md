# Satellite-based Nowcasting of Solar Radiation with AI/ML

This repository contains code and a data pipeline for experimenting with state-of-the-art deep learning models for short-term solar radiation nowcasting, using geostationary satellite images. It includes implementations and training workflows for both **DGMR-SO** and **Earthformer** architectures.

## Data Preperation
Inside the ```/Data``` folder of this repository, you'll find scripts used to prepare, validate, and process the raw satellite data into a format suitable for training the deep learning models described below. This section outlines what each file does and how to use them effectively.

### Step 1. Data processing
The first step is transforming the raw NetCDF (.nc) satellite files into .tfrecords, a format compatible with DGMR-SO and EarthFormer.

#### msgcpp_reduce_subset.py
Creates a geographic subset of the original data, focusing on a specific region that includes the Netherlands, Belgium, France, England, and parts of neighboring countries.  
Usage:
```
python msgcpp_reduce_subset.py
```
Once the subset is generated, split it into a training and testing set. For convenience, place them in folders called: ```raw_train_data``` and ```raw_test_data```.

#### data_preperation_to_tfr.py
Converts the subset .nc files into TFRecord format.
- Uses sliding windows of 20 frames: 4 past frames and 16 future frames.
- Removes windows if:
     - Any of the first 8 frames contain more than 50% darkness (e.g., night).
     - The time sequence is incomplete (e.g., due to missing files).
- Normalizes satellite radiation values using clear-sky radiation (SDS / SDS_CS).
- Requires the helper file tfrecord_shards_for_nowcasting.py. his file is custom made to create TFRecords which are suitable for nowcasting.  

Usage:
```
python data_preperation_tfr.py
```
Update the script with the correct input/output paths and match the height and width to your selected geographic region.

#### select_val_split.py
Splits the training data into training and validation sets using an 80/20 random split. Before execution, make sure the paths are correclty defined.  
Usage:
```
python select_val_split.py
```
The TFRecord files should now be organized in the folders: ```train_data```, ```val_data``` and ```test_data``` in the ```/Data``` folder.

### Step 2. Data quality control 
These scripts validate the integrity and usability of the generated TFRecords and check for any remaining data issues.

#### data_control.py
- Compares the number of raw .nc samples to the number of generated TFRecords.
- Calculates the total dataset size in GB.
- Verifies normalization steps were correctly applied.

Usage:
```
python data_control.py
```

#### check_tfrecords.py
- Confirms none of the TFRecords are corrupted.
- Includes helper functions to:
     - Count how many samples have >50% darkness.
     - Check if any remaining samples contain invalid frames in the first 8 time steps.
 
Usage:  
```
python check_tfrecord.py
```

### Data augmentation
After the data processing and quality control, the ```data_pipleine.py``` file is eventually used by the models to make use of the TFRecord datasets. 
It performs:
- Random cropping
- Flipping (horizontal & vertical)
- Normalization
- Any other augmentations described in the report

## DGMR-SO 
DGMR-SO is an adapted version of the Deep Generative Model for Radar (DGMR) by Google DeepMind, tailored for satellite-based **Surface Solar Irradiance (SSI)** nowcasting. Instead of radar data, this implementation uses geostationary satellite imagery from the SEVIRI instrument aboard the Meteosat-11 satellite, normalized to represent clear-sky index for improved stability. The task is to predict the next 4 hours (16 future timesteps) of SSI based on the past 1 hour (4 timesteps) using rolling windows.

The original paper: https://www.sciencedirect.com/science/article/pii/S0038092X24005619#b0240

### Installation
!IMPORTANT make sure conda, cuda and cuDNN are avaialble/downloaded!  
For Virtual Machine on EUMETSAT, follow these steps: https://confluence.ecmwf.int/display/EWCLOUDKB/EUMETSAT+-+GPU+support

1. Clone the GitHub repository
2. Move into DGMR_SO folder
3. Create conda environment: ```conda create -n dgmr_env python=3.9```
4. Activate conda environment: ```conda activate dgmr_env```
5. Install the required packages:
     - ```pip install tensorflow[and-cuda]```
     - ```pip install matplotlib```
     - ```pip install dm-sonnet```
     - ```pip install pyyaml```
     - ```pip install tensorboard```

After these steps you should be able the execute the next steps.

### Training
Before training an instance of DGMR-SO, make sure the train, validation and test sets are in designated folders in the /Data directory. Furthermore, make sure you are in the DGMR-SO directory and have created an virtual environment desribed in the section above.    
Afterwards typ the following command in the console:  
```
python train.py
```
Or on EUMETSAT: 
```
sudo -E /home/'user'/miniforge3/envs/dgmr_env/bin/python3 train.py
```

This command will execute the full training process for 500.000 steps (or otherwise defined in the train.yml). There is a high change the full 500.00 steps are not necessary. We recommend the use of tensorbaord to watch and evaluate the training process closely.  
When working on a EUMETSAT Virtual Machine and you wat to run tensorboard locally, make sure you connect to the VM as follows:  
```
ssh -L 6006:localhost:6006 'user'@'IP_of_VM'
```
When connected succesfully run the following command to load and view tensorboard on ```localhost:6006```:
```
tensorboard --logdir='directory_to_train_logs' --port=6006 --host=localhost
```

### Inference

## EarthFormer
Earthformer is a space-time transformer architecture designed for Earth system forecasting tasks, such as weather nowcasting and precipitation prediction. Unlike traditional CNN or ConvLSTM-based models, Earthformer leverages spatiotemporal attention mechanisms to effectively capture both short- and long-range dependencies across space and time. The transformer Utilizes axial attention and factorized self-attention to reduce complexity while preserving global context. In this project, Earthformer is adapted to nowcast Surface Solar Irradiance (SSI) using satellite-derived input data.

The original paper: https://www.amazon.science/publications/earthformer-exploring-space-time-transformers-for-earth-system-forecasting

### Installation

### Training

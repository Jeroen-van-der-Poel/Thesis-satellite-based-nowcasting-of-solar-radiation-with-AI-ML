# Satellite-based Nowcasting of Solar Radiation with AI/ML

This repository contains code and a data pipeline for experimenting with state-of-the-art deep learning models for short-term solar radiation nowcasting, using geostationary satellite images. It includes implementations and training workflows for both **DGMR-SO** and **Earthformer** architectures.

## Raw Data Preperation
Inside the ```/RawData``` folder of this repository, you'll find the script used to prepare the raw satellite data with the region of choice. Furthermore this folder contains the dataset class used to prepare the rolling windows.

#### msgcpp_reduce_subset.py
Creates a geographic subset of the original data, focusing on a specific region that includes the Netherlands, Belgium, France, England, and parts of neighboring countries.  
Usage:
```
python msgcpp_reduce_subset.py
```
Once the subset is generated, split it into a training and testing set. For convenience, place them in folders called: ```raw_train_data``` and ```raw_test_data```.

#### netCDFDataset.py
Defines a custom PyTorch Dataset class (NetCDFNowcastingDataset) for loading and preprocessing satellite data stored in NetCDF files. This dataset creates sliding temporal windows of input and target frames for nowcasting tasks.  
Main features:
 - Loads time-sequenced NetCDF files sorted by timestamp.
 - Automatically filters out invalid samples (files with too many dark pixels or broken time intervals).
 - Returns data in the shape required for model input: a concatenated tensor of input and target sequences.


## DGMR-SO 
DGMR-SO is an adapted version of the Deep Generative Model for Radar (DGMR) by Google DeepMind, tailored for satellite-based **Surface Solar Irradiance (SSI)** nowcasting. Instead of radar data, this implementation uses geostationary satellite imagery from the SEVIRI instrument aboard the Meteosat-11 satellite, normalized to represent clear-sky index for improved stability. The task is to predict the next 4 hours (16 future timesteps) of SSI based on the past 1 hour (4 timesteps) using rolling windows.

The original paper: https://www.sciencedirect.com/science/article/pii/S0038092X24005619#b0240

### Installation
!IMPORTANT make sure conda, cuda and cuDNN are avaialble/downloaded!  
For Virtual Machine on EUMETSAT, follow these steps: https://confluence.ecmwf.int/display/EWCLOUDKB/EUMETSAT+-+GPU+support

1. Clone the GitHub repository
2. Move into DGMR_SO folder
3. Create conda environment: ```conda create -n dgmr_env python=3.9 -y```
4. Activate conda environment: ```conda activate dgmr_env```
5. Install the required packages:
     - ```pip install tensorflow[and-cuda]```
     - ```pip install matplotlib```
     - ```pip install dm-sonnet```
     - ```pip install pyyaml```
     - ```pip install tensorboard```

After these steps you should be able the execute the next steps.

### Data preperation

##### data_preperation_to_tfr.py
Converts the subset .nc files into TFRecord format, using the NetCDFNowcastingDataset class.
- Requires the helper file tfrecord_shards_for_nowcasting.py. This file is custom made to create TFRecords which are suitable for nowcasting.
- Creates a 80/20 split for the train and validation dataset.  

Usage:
```
python data_preperation_tfr.py
```
Update the script with the correct input/output paths and match the height and width to your selected geographic region.

##### check_tfrecords.py
A diagnostic and data validation script for checking the integrity and quality of the datasets stored in TFRecord format.  
This script performs the following:
- File corruption detection. 
- Dark frame analysis. Checks the first 8 frames (4 input + 4 target) of each sample for excessively dark pixels.
- NaN/Inf/all-zero detection.
- Shape validation.
- Pixel distribution histogram.
 
Usage:  
```
python check_tfrecord.py
```

#### Step 3. Data augmentation
After the data processing and quality control, the ```data_pipleine.py``` file is eventually used by the models to make use of the TFRecord datasets. 
It performs:
- Random cropping.
- Flipping (horizontal & vertical).
- Normalization.
- Any other augmentations described in the report.

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

This command will execute the full training process for 500.000 steps (or otherwise defined in the train.yml). There is a high change the full 500.00 steps are not necessary. We recommend the use of tensorboard to watch and evaluate the training process closely.  
When working on a EUMETSAT Virtual Machine and you wat to run tensorboard locally, make sure you connect to the VM as follows:  
```
ssh -L 6006:localhost:6006 'user'@'IP_of_VM'
```
When connected succesfully run the following command to load and view tensorboard on ```localhost:6006```:
```
tensorboard --logdir='directory_to_train_logs' --port=6006 --host=localhost
```

## EarthFormer
EarthFormer is a space-time transformer architecture designed for Earth system forecasting tasks, such as weather nowcasting and precipitation prediction. Unlike traditional CNN or ConvLSTM-based models, EarthFormer leverages spatiotemporal attention mechanisms to effectively capture both short- and long-range dependencies across space and time. The transformer Utilizes axial attention and factorized self-attention to reduce complexity while preserving global context. In this project, EarthFormer is adapted to nowcast Surface Solar Irradiance (SSI) using satellite-derived input data.

The original paper: https://www.amazon.science/publications/earthformer-exploring-space-time-transformers-for-earth-system-forecasting

### Installation
!IMPORTANT make sure conda, cuda and cuDNN are avaialble/downloaded!  
For Virtual Machine on EUMETSAT, follow these steps: https://confluence.ecmwf.int/display/EWCLOUDKB/EUMETSAT+-+GPU+support
1. Clone the GitHub repository
2. Move into EarthFormer folder
3. Create conda environment: ```conda create -n ef_env python=3.9 -y```
4. Activate conda environment: ```conda activate ef_env```
5. Install the required packages:
     - ```pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121```
     - ```pip install pytorch_lightning==2.5.1```
     - ```pip install torchmetrics==2.5.1```
     - ```pip install xarray netcdf4 opencv-python earthnet==0.3.9```
     - ```pip install -U -e . --no-build-isolation```

After these steps you should be able the execute the next steps.

### Data preperation

##### data_preperation_to_h5.py
Converts the subset .nc files into h5 format, using the NetCDFNowcastingDataset class.
- Creates a 80/20 split for the train and validation dataset.  

Usage:
```
python data_preperation_to_h5.py
```
Update the script with the correct input/output paths and match the height and width to your selected geographic region.

### Training
Before training an instance of EarthFormer, make sure the train, validation and test sets are in designated folders in the /Data directory. Furthermore, make sure you are in the EarthFormer directory and have created an virtual environment desribed in the section above.    
Afterwards typ the following command in the console: 
```
python train.py --gpu 1 --cfg config/train.yml --ckpt_name last.ckpt --save "version name"
```
Or on EUMETSAT: 
```
sudo -E /home/'user'/miniforge3/envs/ef_env/bin/python3 train.py --gpu 1 --cfg config/train.yml --ckpt_name last.ckpt --save "version name"
```

This command will execute the full training process on the dataset. We recommend the use of tensorboard to watch and evaluate the training process closely.  
When working on a EUMETSAT Virtual Machine and you wat to run tensorboard locally, make sure you connect to the VM as follows:  
```
ssh -L 6006:localhost:6006 'user'@'IP_of_VM'
```
When connected succesfully run the following command to load and view tensorboard on ```localhost:6006```:
```
tensorboard --logdir='directory_to_train_logs' --port=6006 --host=localhost
```

## Comparision
To compare DGMR-SO, EarthFormer and Persistence, we created a evaluation script which evaluates all models on the metrics defined in the report. Furthermore, it generates the images produced and saves them as PNGs.

### Installation
!IMPORTANT make sure conda, cuda and cuDNN are avaialble/downloaded!  
For Virtual Machine on EUMETSAT, follow these steps: https://confluence.ecmwf.int/display/EWCLOUDKB/EUMETSAT+-+GPU+support

1. Move into Comparison folder
2. Create conda environment: ```conda create -n comp_env python=3.9 -y```
3. Activate conda environment: ```conda activate comp_env```
4. Install the required packages:
     - ```pip install tensorflow[and-cuda]```
     - ```pip install matplotlib```
     - ```pip install dm-sonnet```
     - ```pip install pyyaml```
     - ```pip install tensorboard```
     - ```pip install scikit-learn ```
     - ```pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121```
     - ```pip install pytorch_lightning==2.5.1```
     - ```pip install xarray netcdf4 opencv-python earthnet==0.3.9```
     - ```pip install -U -e . --no-build-isolation```
     - ```pip install psutil```
    
 After these steps you should be able the execute the next steps.

### Execution
Make sure the path in the ```compare_models.py``` script are correclty defined. Afterwards run:  
``` 
python compare_models.py
```  
The images and metric scores will be saved in the /vis folder.


# import tensorflow.compat.v1 as tf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Data.data_pipeline import Dataset
from model.dgmr import DGMR
from utils.losses import Loss_hing_disc, Loss_hing_gen
import os
from utils.utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print('GPU:', gpus)

cfg = read_yaml(Path('./config/' + 'train_0' + '.yml'))

MODEL_NAME = cfg['model_identification']['model_name']
MODEL_VERSION = cfg['model_identification']['model_version']
ROOT = get_project_root()
CHECKPOINT_DIR = ROOT / 'Checkpoints' / \
    (str(MODEL_NAME) + '_v' + str(MODEL_VERSION))
make_dirs([CHECKPOINT_DIR])

training_steps = cfg['model_params']['steps']

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(18)
# tf.config.set_soft_device_placement(True)
# gpu_devices_list = tf.config.list_physical_devices('GPU')

batch_size = 16
train_data,train_dataset_aug = Dataset(Path('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data'), batch_size=batch_size)
val_data,val_data_val = Dataset(Path('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data'), batch_size=batch_size)

train_writer = tf.summary.create_file_writer(str(ROOT / "logs" / (str(MODEL_NAME) + '_v' + str(MODEL_VERSION)) / "train/"))
print("Log directory:", str(ROOT / "logs" / (str(MODEL_NAME) + '_v' + str(MODEL_VERSION)) / "train/"))
prof_dir = str(ROOT / "logs" / (str(MODEL_NAME) + '_v' + str(MODEL_VERSION)) / "profiler/")
# profiler_writer = tf.summary.create_file_writer(prof_dir)

# INIT MODEL
disc_optimizer = Adam(learning_rate=2E-4, beta_1=0.0, beta_2=0.999)
gen_optimizer = Adam(learning_rate=1E-5, beta_1=0.0, beta_2=0.999)
loss_hinge_gen = Loss_hing_gen()
loss_hinge_disc = Loss_hing_disc()

# with strategy.scope() :
tf.keras.backend.clear_session()
my_model = DGMR(lead_time=240, time_delta=15)
my_model.trainable = True
my_model.compile(gen_optimizer, disc_optimizer, loss_hinge_gen, loss_hinge_disc)

ckpt = tf.train.Checkpoint(generator=my_model.generator_obj, discriminator=my_model.discriminator_obj, generator_optimizer=my_model.gen_optimizer, discriminator_optimizer=my_model.disc_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=100)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

gen_loss, disc_loss = my_model.fit(train_dataset_aug, val_data, steps=training_steps, callbacks=[train_writer, ckpt_manager, ckpt, prof_dir])
tf.keras.backend.clear_session()
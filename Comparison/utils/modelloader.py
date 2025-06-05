from DGMR_SO.model.dgmr import DGMR
from EarthFormer.train import CuboidPLModule
# import tensorflow as tf
import torch

# def load_dgmr_model(checkpoint_path):
#     dgmr = DGMR(lead_time=240, time_delta=15)
#     ckpt = tf.train.Checkpoint(generator=dgmr.generator_obj)
#     ckpt.restore(checkpoint_path)
#     return dgmr.generator_obj

def load_earthformer_model(state_dict_path, cfg_file_path, save_dir="tmp_netcdf", test_dataset=None):
    total_num_steps = CuboidPLModule.get_total_num_steps(
        epoch=100,
        num_samples=len(test_dataset),
        total_batch_size=32,
    )
    model = CuboidPLModule(total_num_steps=total_num_steps, oc_file=cfg_file_path, save_dir=save_dir)
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
    model.torch_nn_module.load_state_dict(state_dict)
    model.eval()
    return model.torch_nn_module 

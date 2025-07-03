from DGMR_SO.model.dgmr import DGMR
from keras.optimizers import Adam
from DGMR_SO.utils.losses import Loss_hing_disc, Loss_hing_gen
import tensorflow as tf
import torch
import numpy as np

class DGMRWrapper:
    def __init__(self, checkpoint_path, seed=42):
        self.model = self._load_model(checkpoint_path)
        self.crop_height = 256
        self.crop_width = 256
        self.lambda_reg = 1
        self.seed = seed

    def _load_model(self, checkpoint_path):
        disc_optimizer = Adam(learning_rate=2E-4, beta_1=0.0, beta_2=0.999)
        gen_optimizer = Adam(learning_rate=1E-5, beta_1=0.0, beta_2=0.999)
        model = DGMR(lead_time=240, time_delta=15)
        model.compile(gen_optimizer, disc_optimizer, Loss_hing_gen(), Loss_hing_disc())
        ckpt = tf.train.Checkpoint(generator=model.generator_obj,
                                   discriminator=model.discriminator_obj,
                                   generator_optimizer=model.gen_optimizer,
                                   discriminator_optimizer=model.disc_optimizer)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        return model

    def __call__(self, inputs, targets=None):
        # inputs: torch tensor, shape [B, T_in, H, W, 1]
        inputs_np = inputs.detach().cpu().numpy()
        if targets is None:
            targets_np = inputs_np.copy()  
        else:
            targets_np = targets.detach().cpu().numpy()

        # Convert to TF tensors
        inputs_tf = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
        targets_tf = tf.convert_to_tensor(targets_np, dtype=tf.float32)
        inputs_tf, targets_tf_crop, y_coords, x_coords = self._random_crop_with_coords(inputs_tf, targets_tf, self.crop_height, self.crop_width)

        # Run inference
        outputs_tf = self.model.generator_obj(inputs_tf, is_training=True)

        # Reconstruct full-size output
        B, T_out, _, _, C = outputs_tf.shape
        H_orig = inputs.shape[2]
        W_orig = inputs.shape[3]
        full_output = tf.Variable(tf.zeros((B, T_out, H_orig, W_orig, C), dtype=outputs_tf.dtype))

        resized_outputs = tf.image.resize(outputs_tf, size=(H_orig, W_orig), method='bilinear')
        full_output = resized_outputs  

        outputs_np = full_output.numpy()
        outputs_np_cropped = outputs_tf.numpy()
        targets_np = targets_tf.numpy()
        target_np_cropped = targets_tf_crop.numpy()

        return (
            torch.tensor(outputs_np, dtype=torch.float32),
            torch.tensor(targets_np, dtype=torch.float32),
            torch.tensor(outputs_np_cropped, dtype=torch.float32),
            torch.tensor(target_np_cropped, dtype=torch.float32),
            y_coords,
            x_coords 
        )    
    
    def _random_crop_with_coords(self, input_tensor, label_tensor, crop_height, crop_width):
        B = input_tensor.shape[0]
        T = input_tensor.shape[1]
        H = input_tensor.shape[2]
        W = input_tensor.shape[3]
        C = input_tensor.shape[4]

        cropped_inputs = []
        cropped_labels = []
        y_coords = []
        x_coords = []

        rng = np.random.default_rng(self.seed) 

        for i in range(B):
            y = rng.integers(0, H - crop_height + 1)
            x = rng.integers(0, W - crop_width + 1)
            y_coords.append(y)
            x_coords.append(x)

            cropped_input = input_tensor[i:i+1, :, y:y+crop_height, x:x+crop_width, :]
            cropped_label = label_tensor[i:i+1, :, y:y+crop_height, x:x+crop_width, :]
            cropped_inputs.append(cropped_input)
            cropped_labels.append(cropped_label)

        cropped_inputs = tf.concat(cropped_inputs, axis=0)
        cropped_labels = tf.concat(cropped_labels, axis=0)

        return cropped_inputs, cropped_labels, y_coords, x_coords
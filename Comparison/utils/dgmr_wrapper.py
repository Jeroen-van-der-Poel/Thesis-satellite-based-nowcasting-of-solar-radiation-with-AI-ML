from DGMR_SO.model.dgmr import DGMR
from keras.optimizers import Adam
from DGMR_SO.utils.losses import Loss_hing_disc, Loss_hing_gen
import tensorflow as tf
import torch

class DGMRWrapper:
    def __init__(self, checkpoint_path):
        self.model = self._load_model(checkpoint_path)
        self.crop_height = 256
        self.crop_width = 256
        self.lambda_reg = 1

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
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            print("DGMR-SO checkpoint restored!")
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
        inputs_tf, targets_tf = self.model.random_crop_images(inputs_tf, targets_tf, self.crop_height, self.crop_width)

        # Run inference
        outputs_tf = self.model.generator_obj(inputs_tf, is_training=False)
        outputs_np = outputs_tf.numpy()

        print("Output min/max:", outputs_tf.numpy().min(), outputs_tf.numpy().max())
        print("Target min/max:", targets_tf.numpy().min(), targets_tf.numpy().max())

        # Optional: scale predictions
        # outputs_np = outputs_np * 1000.0

        # Return both prediction and cropped targets for evaluation
        return torch.tensor(outputs_np, dtype=torch.float32), torch.tensor(targets_tf.numpy(), dtype=torch.float32)


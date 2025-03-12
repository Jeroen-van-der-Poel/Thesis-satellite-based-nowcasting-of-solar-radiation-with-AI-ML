# import sonnet as snt
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt
import time
import datetime

class DGMR(tf.keras.Model):
    def __init__(self, lead_time=240, time_delta=15) -> None:
        super(DGMR, self).__init__()
        self.strategy = None
        self.global_step = 0
        self.generator_obj = Generator(lead_time=lead_time, time_delta=time_delta)
        self.discriminator_obj = Discriminator()

    @tf.function
    def __call__(self, tensor, is_training=True):
        return self.generator_obj(tensor, is_training=is_training)

    def compile(self, gen_optimizer, disc_optimizer, gen_loss, disc_loss):
        # super(DGMR, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss

    def fit(self, dataset,  steps=2, callbacks=[]):
        train_writer = callbacks[0]
        ckpt_manager = callbacks[1]
        ckpt = callbacks[2]
        disc_loss_l = []
        gen_loss_l = []

        '''if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')'''
        dataset = iter(dataset)
        num_batches = self.global_step

        '''start_profiling_step = 40
        stop_profiling_step = 90'''

        for step in range(steps):
            # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):

            '''if step == start_profiling_step:
              tf.profiler.experimental.start(logdir=callbacks[3])
            if step == stop_profiling_step:
              tf.profiler.experimental.stop(save=True)'''

            batch_inputs1, batch_targets1, _,targ_mask1 = next(dataset)
            batch_inputs2, batch_targets2, _,targ_mask2 = next(dataset)

            temp_time = time.time()
            # self.train_step( batch_inputs, batch_targets)
            gen_loss, disc_loss = self.distributed_train_step(batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2)

            if num_batches < 50:
                tf.print("Time per epoch: ", time.time() - temp_time)

            # Loss
            disc_loss_l.append(disc_loss)
            gen_loss_l.append(gen_loss)

            num_batches += 1

            if step and (step % 100 == 0):
                tf.print("Gen Loss: ", gen_loss.numpy(),
                         " Disc Loss: ", disc_loss.numpy())

                # val loss
                # TODO evaluate with evaluation metrics

            if step and (step % 2000 == 0):
                ckpt_save_path = ckpt_manager.save()

            if step and (step % 5 == 0):
                with train_writer.as_default():
                    tf.summary.scalar("Gen_loss", gen_loss, step=step)
                    tf.summary.scalar("Disc_loss", disc_loss, step=step)

        # tf.profiler.experimental.stop()

        return gen_loss_l, disc_loss_l
    # (input_signature=[tf.TensorSpec(shape=[2, 4, 256, 256, 1], dtype=tf.float32), tf.TensorSpec(shape=[2, 2, 256, 256, 1], dtype=tf.float32)])

    @tf.function
    def distributed_train_step(self, batch_inputs1, batch_targets1, targ_mask1,  batch_inputs2, batch_targets2, targ_mask2):
        #tf.print("Debugging: before run")
        per_replica_g_losses, per_replica_d_losses = self.strategy.run(self.train_step, args=(batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2))
        #tf.print("Debugging: after run")

        #tf.print("Debugging: per_replica_g_losses -> ", per_replica_g_losses)
        #tf.print("Debugging: per_replica_d_losses -> ",per_replica_d_losses)

        total_g_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_g_losses, axis=0)
        total_d_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_d_losses, axis=0)

        #tf.print("Debugging: total_g_loss -> ", total_g_loss)
        #tf.print("Debugging: total_d_loss -> ", total_d_loss)
        return total_g_loss, total_d_loss

    @tf.function
    def train_step(self, batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2):
        self.disc_step(batch_inputs1, batch_targets1, targ_mask1)
        disc_loss = self.disc_step(batch_inputs2, batch_targets2, targ_mask2)

        #self.gen_step(batch_inputs2, batch_targets2, targ_mask2)
        #self.gen_step(batch_inputs2, batch_targets2, targ_mask2)
        gen_loss = self.gen_step(batch_inputs2, batch_targets2, targ_mask2)

        return gen_loss, disc_loss

    def disc_step(self, batch_inputs, batch_targets, targ_mask, is_training=True):
        #tf.print("Debugging: disc_step: ", "Disc step has started", str(datetime.datetime.now()))
        with tf.GradientTape() as disc_tape:
            batch_predictions = self.generator_obj(batch_inputs, is_training=is_training)
            '''batch_predictions = tf.where(
                tf.equal(targ_mask, True), batch_predictions, -1)'''
            gen_sequence = tf.concat([batch_inputs[..., :1], batch_predictions], axis=1)
            real_sequence = tf.concat([batch_inputs[..., :1], batch_targets], axis=1)
            concat_inputs = tf.concat([real_sequence, gen_sequence], axis=0)

            concat_outputs = self.discriminator_obj(concat_inputs, is_training=is_training)

            score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
            disc_loss = self.disc_loss(score_generated, score_real)

        disc_grads = disc_tape.gradient(disc_loss, self.discriminator_obj.trainable_variables)

        # Aggregate the gradients from the full batch.
        '''replica_ctx_disc = tf.distribute.get_replica_context()
        disc_grads = replica_ctx_disc.all_reduce(
            tf.distribute.ReduceOp.MEAN, disc_grads)'''

        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator_obj.trainable_variables))

        #tf.print("Debugging: disc_loss -> ", disc_loss)
        #tf.print("Debugging: disc_step: ", "Disc step ended", str(datetime.datetime.now()))
        return disc_loss

    def gen_step(self, batch_inputs, batch_targets, targ_mask, is_training=True):
        #tf.print("Debugging: gen_step: ", "Gen step has started", str(datetime.datetime.now()))
        with tf.GradientTape() as gen_tape:
            num_samples_per_input = 3  # FIXME it was 6.
            gen_samples = [self.generator_obj(batch_inputs, is_training=is_training)
                           for _ in range(num_samples_per_input)]
            
            grid_cell_reg = grid_cell_regularizer(tf.stack(gen_samples, axis=0), batch_targets)
            '''gen_samples = [tf.where(
                tf.equal(targ_mask, True), gen_sample, -1) for gen_sample in gen_samples]'''
            gen_sequences = [tf.concat([batch_inputs[..., :1], x], axis=1)
                             for x in gen_samples]
            real_sequence = tf.concat([batch_inputs[..., :1], batch_targets], axis=1)

            generated_scores = []
            for g_seq in gen_sequences:
                concat_inputs = tf.concat([real_sequence, g_seq], axis=0)
                concat_outputs = self.discriminator_obj(
                    concat_inputs, is_training=is_training)
                score_real, score_generated = tf.split(
                    concat_outputs, 2, axis=0)
                generated_scores.append(score_generated)

            gen_disc_loss = self.gen_loss(tf.concat(generated_scores, axis=0))
            gen_loss = gen_disc_loss + 15. * grid_cell_reg

        gen_grads = gen_tape.gradient(gen_loss, self.generator_obj.trainable_variables)

        # Aggregate the gradients from the full batch.
        '''replica_ctx_disc = tf.distribute.get_replica_context()
        disc_grads = replica_ctx_disc.all_reduce(
            tf.distribute.ReduceOp.MEAN, disc_grads)'''

        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator_obj.trainable_variables))

        #tf.print("Debugging: gen_loss -> ", gen_loss)
        #tf.print("Debugging: gen_step: ", "Gen step has ended", str(datetime.datetime.now()))
        return gen_loss


def grid_cell_regularizer(generated_samples, batch_targets):
    """Grid cell regularizer.

    Args:
      generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
      batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

    Returns:
      loss: A tensor of shape [batch_size].
    """
    gen_mean = tf.reduce_mean(generated_samples, axis=0)
    weights = tf.clip_by_value(batch_targets, 0.0, 24.0)
    loss = tf.reduce_mean(tf.math.abs(gen_mean - batch_targets) * weights)
    return loss
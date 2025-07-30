import tensorflow as tf

class Loss_hing_disc():
    def __init__(self) -> None:
        pass

    def __call__(self, score_generated, score_real):
        """Discriminator hinge loss."""
        l1 = tf.nn.relu(1. - score_real)
        loss = tf.reduce_mean(l1) 
        l2 = tf.nn.relu(1. + score_generated)
        loss += tf.reduce_mean(l2)  
        tf.print("Debugging: Disc Loss: ", loss)
        return loss

class Loss_hing_gen():
    def __init__(self) -> None:
        pass

    def __call__(self, score_generated):
        """Generator hinge loss."""
        loss = - \
            tf.reduce_mean(
                score_generated) 
        tf.print("Debugging, Gen Loss: ", loss)
        return loss
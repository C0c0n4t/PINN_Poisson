import tensorflow as tf
from tensorflow.python import keras
from math import pi
import numpy as np
from data_gen import DataGenerator, real_u1

tf_pi = tf.constant(pi)

@tf.function
def custom_activation2(x):
    return tf.sin(5*x)

@tf.function
def custom_activation(x):
    return tf.sin(x)


class pinnModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(units=32, activation=custom_activation2)
        self.denese2 = keras.layers.Dense(units=32, activation=custom_activation)

    def f(self, x, y):
        return -2 * tf_pi * tf_pi * tf.sin(tf_pi * y) * tf.sin(tf_pi * x)

    # data - inside, ic - inital conditon
    def train_step(self, data, ic, k):
        with tf.GradientTape() as tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape1:
                tape1.watch(data)
                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape2:
                    tape2.watch(data)
                    u = self(data, training=True)
                grad_u = tape2.gradient(u, data)
                du_dx = grad_u[..., 0]
                du_dy = grad_u[..., 1]
                del tape2

            d2u_dx2 = tape1.gradient(du_dx, data)[..., 0]
            d2u_dy2 = tape1.gradient(du_dy, data)[..., 1]
            del tape1

            x = data[..., 0]
            y = data[..., 1]
            ode_loss = d2u_dx2 + d2u_dy2 - self.f(x, y)
            IC_loss = self(ic) - tf.zeros((len(ic), 1))

            loss = tf.reduce_mean(tf.square(ode_loss)) + \
                k * tf.reduce_mean(tf.square(IC_loss))

        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grad, self.trainable_variables))
        del tape

        # Return a dict mapping metric names to current value
        return {}
    

model = pinnModel()

model.compile(optimizer="adam", loss="mse", metrics=[])

dg = DataGenerator((0, 2), (0, 2), model, real_u1)
model.fit(dg.area_pairs((50, 50)), dg.inner_pairs((50, 50)), epochs=3)

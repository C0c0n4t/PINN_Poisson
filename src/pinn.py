import os
from math import pi
import tensorflow as tf


def model1():
    @tf.function
    def custom_activation(x):
        return tf.sin(x)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((2,)),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=1),
        ]
    )

    return model


tf_pi = tf.constant(pi)


class PINNModel:
    """
    --------------------------
    :param model: model() from models.py
    :param optm: optm for model
    :param str initial_weights: path to initial weights
    """

    def __init__(self, model, optm, initial_weights: str = "") -> None:
        self._model = model
        self._optm = optm

        self._init_w = initial_weights

        self._koef = None
        self._ic: tf.Variable = None
        self._bc: tf.Variable = None

        self._model.compile(optimizer=self._optm, loss="mean_squared_error")
        if self._init_w != "":
            self._model.load_weights(self._init_w)

    # @tf.function
    @staticmethod
    def f(x, y):
        return -2 * tf_pi * tf_pi * tf.sin(tf_pi * y) * tf.sin(tf_pi * x)

    @tf.function
    def _ode(self):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self._ic)
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape1:
                tape1.watch(self._ic)
                u = self._model(self._ic)
            grad_u = tape1.gradient(u, self._ic)
            du_dx = grad_u[..., 0]
            du_dy = grad_u[..., 1]
            del tape1

        d2u_dx2 = tape.gradient(du_dx, self._ic)[..., 0]
        d2u_dy2 = tape.gradient(du_dy, self._ic)[..., 1]
        del tape

        x = self._ic[..., 0]
        y = self._ic[..., 1]
        ode_loss = d2u_dx2 + d2u_dy2 - self.f(x, y)
        IC_loss = self._model(self._bc) - tf.zeros((len(self._bc), 1))

        return tf.reduce_mean(tf.square(ode_loss)) + self._koef * tf.reduce_mean(tf.square(IC_loss))

    @tf.function
    def _train_cycle(self, epochs, loss, eprint):
        for itr in tf.range(0, epochs):
            with tf.GradientTape() as tape:
                train_loss = self._ode()
                # TODO: tf.summary
                # train_loss_record.append(train_loss)

            grad_w = tape.gradient(train_loss, self._model.trainable_variables)
            self._optm.apply_gradients(
                zip(grad_w, self._model.trainable_variables))
            del tape

            if itr % eprint == 0:
                # USE TF.PRINT()!!!
                tf.print("epoch:", itr, "loss:", train_loss)
                if train_loss < loss:
                    break

    def train(self, koef, inner, border, epochs, loss, eprint):
        self._koef = tf.constant(koef, dtype=tf.float32)
        self._ic = tf.Variable(inner)
        self._bc = tf.Variable(border)
        self._train_cycle(epochs, loss, eprint)

    def predict(self, area):
        return self._model.predict(area)

    def load_weights(self, path):
        try:
            self._model.load_weights(path)
        except:
            print("No such file")

    def save(self, path: str):
        if not os.path.isfile(path):
            os.mknod(path)
        self._model.save_weights(path)

    def reset_weights(self):
        self._model.set_weights(self._init_w)

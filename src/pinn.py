import os
from math import pi
import tensorflow as tf


tf_pi = tf.constant(pi)


class PINNModel:
    """
    --------------------------
    :param model: model() from models.py
    :param optm: optm for model
    :param str initial_weights: path to initial weights
    """

    def __init__(self, model: tf.keras.models.Sequential, optm, initial_weights: str="") -> None:
        self._model: tf.keras.models.Sequential = model
        self._optm = optm

        self._init_w = initial_weights
        self._koef = None
        self._ic = None
        self._bc = None

        self._model.compile(optimizer=self._optm, loss="mean_squared_error")
        if self._init_w != "":
            self._model.load_weights(self._init_w)

    # @tf.function
    def f(self, x, y):
        print(x.shape, y.shape)
        return -2 * tf_pi * tf_pi * tf.sin(tf_pi * y) * tf.sin(tf_pi * x)

    @tf.function
    def _ode(self, ic, bc):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(ic)
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape1:
                tape1.watch(ic)
                u = self._model(ic)
            grad_u = tape1.gradient(u, ic)
            du_dx = grad_u[..., 0]
            du_dy = grad_u[..., 1]
            del tape1

        d2u_dx2 = tape.gradient(du_dx, ic)[..., 0]
        d2u_dy2 = tape.gradient(du_dy, ic)[..., 1]
        del tape

        x = ic[..., 0]
        y = ic[..., 1]
        print(x.shape, y.shape)
        ode_loss = d2u_dx2 + d2u_dy2 - self.f(x, y)
        IC_loss = self._model(bc) - tf.zeros((len(bc), 1))

        return tf.reduce_mean(tf.square(ode_loss)) + self._koef * tf.reduce_mean(tf.square(IC_loss))

    @tf.function
    def _train_cycle(self, epochs, loss, eprint):
        for itr in tf.range(0, epochs):
            with tf.GradientTape() as tape:
                train_loss = self._ode(self._ic, self._bc)
                # TODO: tf.summary
                # train_loss_record.append(train_loss)

            grad_w = tape.gradient(train_loss, self._model.trainable_variables)
            self._optm.apply_gradients(
                zip(grad_w, self._model.trainable_variables))
            del tape

            if itr % eprint == 0:
                # USE TF.PRINT()!!!
                tf.print("epoch:", itr, "loss:", train_loss)  # .numpy())
                if train_loss < loss:
                    break

    def train(self, koef, train_coord, border, epochs, loss, eprint):
        self._koef = tf.constant(koef, dtype=tf.float32)
        self._ic = train_coord
        self._bc = border
        self._train_cycle(epochs, loss, eprint)

    def load_weights(self, path):
        try:
            self._model.load_weights(path)
        except:
            print("No such file")

    def reset_weights(self):
        self._model.set_weights(self._initial_weights)

    def save(self, path: str):
        if not os.path.isfile(path):
            os.mknod(path)
        self._model.save_weights(path)

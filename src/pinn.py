import os
import time
from math import pi
import tensorflow as tf


def model1():
    @tf.function
    def custom_activation2(x):
        return tf.sin(5*x)

    @tf.function
    def custom_activation(x):
        return tf.sin(x)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((2,)),
            tf.keras.layers.Dense(units=32, activation=custom_activation2),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=1),
        ]
    )

    model.summary()
    return model


def model2():
    @tf.function
    def custom_activation(x):
        return tf.sin(x)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((2,)),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=32, activation=custom_activation),
            tf.keras.layers.Dense(units=1),
        ]
    )

    model.summary()
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

        self._train_loss = None

        self._EPOCHS = 5000
        self._EPRINT = 500
        self._koef = None
        self._ic: tf.Variable = None
        self._bc: tf.Variable = None

        self._model.compile(optimizer=self._optm, loss="mean_squared_error")

        if initial_weights != "":
            self.load_weights(initial_weights)
        self._init_w = self._model.get_weights()

    # @tf.function
    def f(self, x, y):
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
    def _train_cycle(self):
        for itr in tf.range(0, self._EPOCHS):
            with tf.GradientTape() as tape:
                train_loss = self._ode()
                # TODO: tf.summary
                # train_loss_record.append(train_loss)

            if itr % self._EPRINT == 0:
                # USE TF.PRINT()!!!
                tf.print("epoch:", itr, "loss:", train_loss)

            grad_w = tape.gradient(train_loss, self._model.trainable_variables)
            self._optm.apply_gradients(
                zip(grad_w, self._model.trainable_variables))
            del tape

    def fit(self, koef, inner, border, EPOCHS, EPRINT):
        start = time.time()

        self._koef = tf.constant(koef, dtype=tf.float32)
        self._ic = tf.Variable(inner)
        self._bc = tf.Variable(border)
        self._EPOCHS = tf.Variable(EPOCHS)
        self._EPRINT = tf.Variable(EPRINT)
        # self._train_loss = []
        self._train_cycle()

        print(f"Time past {time.time() - start}\n")
        # return self._train_loss

    def predict(self, area):
        return self._model.predict(area)

    def load(self, path):
        try:
            self._model.load_weights(path)
        except Exception as e:
            print("No such file")
            print(e)

    def save(self, path: str):
        if not os.path.isfile(path):
            os.mknod(path)
        self._model.save_weights(path)

    def reset(self):
        self._model.set_weights(self._init_w)

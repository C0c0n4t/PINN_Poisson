import os
import time
from math import pi
import tensorflow as tf
from typing import Iterable, Callable


def optm1():
    boundaries = [5000, 15000, 50000]
    values = [1e-3, 1e-4, 1e-5, 1e-6]

    lr_sched = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    return tf.keras.optimizers.Adam(learning_rate=lr_sched)


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

    return model


def model3():
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

    return model


@tf.function
def de1(x, y):
    tf_pi = tf.constant(pi)
    return tf.multiply(-2., tf_pi) * tf_pi * tf.sin(tf_pi * y) * tf.sin(tf_pi * x)


@tf.function
def ic1(xy):
    return tf.zeros((xy.shape[0], 1))


class PINNModel:
    """
    --------------------------
    :param model: model() from models.py
    :param optm: optm for model
    :param str initial_weights: path to initial weights
    """

    # TODO : pass f and icf as parameters
    def __init__(self,
                 model, optm, lossf="mse", initial_weights: str = "", de: Callable = de1, ic: Callable = ic1) -> None:
        self._train_loss = None

        self._EPOCHS = None
        self._koef = None

        self._ic = ic
        self._de = de

        self._ins: tf.Variable | None = None
        self._bor: tf.Variable | None = None

        self._model = model
        # self._optm = optm
        self._model.compile(optimizer=optm, loss=lossf)

        self._init_path = initial_weights
        if os.path.exists(self._init_path):
            self._model.load_weights(self._init_path)

        # self.cdir = '../model/checkpoints'
        self.best_loss = tf.Variable(1e3, dtype=tf.float32)
        self.checkpoint = tf.train.Checkpoint(model=model)

    @tf.function
    def _train_step(self):
        with tf.GradientTape() as tape:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape2:
                tape2.watch(self._ins)
                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape1:
                    tape1.watch(self._ins)
                    u = self._model(self._ins)
                grad_u = tape1.gradient(u, self._ins)
                du_dx = grad_u[..., 0]
                du_dy = grad_u[..., 1]
                del tape1

            d2u_dx2 = tape2.gradient(du_dx, self._ins)[..., 0]
            d2u_dy2 = tape2.gradient(du_dy, self._ins)[..., 1]
            del tape2

            ode_loss = d2u_dx2 + d2u_dy2 - \
                self._de(self._ins[..., 0], self._ins[..., 1])
            IC_loss = self._model(self._bor) - self._ic(self._bor)

            loss = tf.reduce_mean(tf.square(ode_loss)) + \
                self._koef * tf.reduce_mean(tf.square(IC_loss))
            # tf.print("loss", loss)

        # save model with best weights
        if tf.less(loss, self.best_loss):
            self.best_loss.assign(loss)
            self.checkpoint.write("../models/checkpoints/ckpt")

        grad_w = tape.gradient(loss, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(
            zip(grad_w, self._model.trainable_variables))
        # self._optm.apply_gradients(
        #     zip(grad_w, self._model.trainable_variables))
        del tape

    def _train_loop(self):
        for _ in tf.range(0, self._EPOCHS):
            self._train_step()

    def fit(self, koef, inner, border, EPOCHS: int,
            LOSS: int | None = None, ERROR: int | None = None):
        start = time.time()

        # measuring time
        self._koef = tf.constant(koef, dtype=tf.float32)
        self._ins = tf.Variable(inner)
        self._bor = tf.Variable(border)

        self._EPOCHS = tf.Variable(EPOCHS)
        self._train_loop()

        latest_checkpoint = tf.train.latest_checkpoint('../models/checkpoints')
        self.checkpoint.restore(latest_checkpoint)
        #

        print(f"Time past {time.time() - start}\n")

    def predict(self, area):
        return self._model.predict(area)

    def loss(self, koef, inner, border):
        inner = tf.Variable(inner)
        border = tf.Variable(border)
        koef = tf.constant(float(koef))

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(inner)
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape1:
                tape1.watch(inner)
                u = self._model(inner)
            grad_u = tape1.gradient(u, inner)
            du_dx = grad_u[..., 0]
            du_dy = grad_u[..., 1]
            del tape1

        d2u_dx2 = tape.gradient(du_dx, inner)[..., 0]
        d2u_dy2 = tape.gradient(du_dy, inner)[..., 1]
        del tape

        x = inner[..., 0]
        y = inner[..., 1]
        ode_loss = d2u_dx2 + d2u_dy2 - self._de(x, y)
        IC_loss = self._model(border) - tf.zeros((border.shape[0], 1))

        return (tf.reduce_mean(tf.square(ode_loss)) + koef * tf.reduce_mean(tf.square(IC_loss))).numpy()

    def load(self, path: str):
        try:
            self._model.load_weights(path)
        except Exception as e:
            print("No such file")

    def save(self, path: str):
        self._model.save_weights(path)

    # not working
    def reset(self):
        # self._model.load_weights(self._init_path)

        # # Create a new optimizer with the same configuration as the original optimizer
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # # Set the weights of the new optimizer to the weights of the original optimizer
        # optimizer.set_weights(new_model.optimizer.get_weights())

        # # Compile the new model with the new optimizer
        # self_model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        pass

import numpy as np
import numpy.typing as npt
import json
import os
import re
from typing import Callable


def real_u1(area):
    area = np.array(area)
    if len(area.shape) >= 3:
        x = area[0]
        y = area[1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    else:
        return np.array([np.sin(np.pi * x) * np.sin(np.pi * y) for x, y in area])


def get_data(sess: int) -> dict:
    with open('data.json', 'r') as file:
        data = json.load(file)
    data["x"] = tuple(data["x"])
    data["y"] = tuple(data["y"])
    last_sess = 0
    sess_exists = False
    for f in os.listdir("../models"):
        if re.match("^s\d+$", f):
            if int(f[f.find("s") + 1:]) == sess:
                sess_exists = True
            last_sess = max(last_sess, int(
                f[f.find("s") + 1:]))
    data["last_session"] = last_sess
    data["session_exists"] = sess_exists
    return data


class DataGenerator:
    def __init__(self, x_limits: tuple, y_limits: tuple, predict: Callable | None, real: Callable):
        self._xlim = x_limits
        self._ylim = y_limits
        self._predict = predict
        self._real = real

    @staticmethod
    def name(koef, sess) -> str:
        return f"../models/s{sess}/model{koef}.weights.h5"

    @staticmethod
    def init_name(sess) -> str:
        return f"../models/s{sess}/initial.weights.h5"

    def update_predict(self, predict: Callable):
        self._predict = predict

    def inner_pairs(self, grid):
        x = np.linspace(self._xlim[0], self._xlim[1],
                        grid[0], dtype=np.float32)[1:-1]
        y = np.linspace(self._ylim[0], self._ylim[1],
                        grid[1], dtype=np.float32)[1:-1]

        return self.__mesh_to_pairs(np.meshgrid(x, y))

    def border_pairs(self, grid):
        x = np.linspace(self._xlim[0], self._xlim[1],
                        grid[0], dtype=np.float32)
        y = np.linspace(self._ylim[0], self._ylim[1],
                        grid[1], dtype=np.float32)

        x_first = np.full(grid[0], x[0])
        x_last = np.full(grid[0], x[-1])
        y_first = np.full(grid[1], y[0])
        y_last = np.full(grid[1], y[-1])
        border = np.concatenate((np.column_stack((x_first, x)), np.column_stack(
            (x_last, x)), np.column_stack((y, y_first))[1:-1], np.column_stack((y, y_last))[1:-1]))
        return border

    def area_pairs(self, grid):
        return np.concatenate((self.border_pairs(grid), self.inner_pairs(grid)), axis=0)

    def real_pairs(self, grid):
        return self._real(self.area_pairs(grid))

    def prediction_pairs(self, grid):
        if not (self._predict is None):
            return self._predict(self.area_pairs(grid))

    def plot_area(self, grid):
        x = np.linspace(self._xlim[0], self._xlim[1], grid[0])
        y = np.linspace(self._ylim[0], self._ylim[1], grid[1])
        x, y = np.meshgrid(x, y)
        pred_coord = []
        for _x in x[0]:
            for _y in x[0]:
                pred_coord.append([_x, _y])
        pred_coord = np.array(pred_coord)
        true_u = self._real((x, y))

        if not (self._predict is None):
            pred_u = self._predict(pred_coord).ravel().reshape(grid)
        else:
            raise Exception(
                "В генераторе для обучения вы пытаетесь использовать predict")
        return (x, y, true_u, pred_u)

    @staticmethod
    def __mesh_to_pairs(meshgrid: list[np.ndarray]) -> npt.NDArray:
        return np.column_stack((meshgrid[0].flatten(), meshgrid[1].flatten()))

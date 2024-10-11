import numpy as np
import numpy.typing as npt
from typing import Callable


class DataGenerator:
    def __init__(self, x_limits, y_limits, predict: Callable, real: Callable):
        self._xlim = x_limits
        self._ylim = y_limits
        self._predict = predict
        self._real = real

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
        pred_u = self._predict(pred_coord).ravel().reshape(grid)
        return (x, y, true_u, pred_u)

    @staticmethod
    def __mesh_to_pairs(meshgrid: list[np.ndarray]) -> npt.NDArray:
        return np.column_stack((meshgrid[0].flatten(), meshgrid[1].flatten()))

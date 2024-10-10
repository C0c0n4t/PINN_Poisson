import numpy as np
import numpy.typing as npt
from typing import Callable


class DataGenerator:
    def __init__(self, x_limits, y_limits, predict: Callable, real: Callable):
        self._xlim = x_limits
        self._ylim = y_limits
        self._predict = predict
        self._real = real

    def inner_pairs(self, gs):
        x = np.linspace(self._xlim[0], self._xlim[1],
                        gs[0], dtype=np.float32)[1:-1]
        y = np.linspace(self._ylim[1][0], self._ylim[1],
                        gs[1], dtype=np.float32)[1:-1]

        return self.__mesh_to_pairs(np.meshgrid(x, y))

    def border_pairs(self, gs):
        x = np.linspace(self._xlim[0], self._xlim[1],
                        gs[0], dtype=np.float32)
        y = np.linspace(self._ylim[0], self._ylim[1],
                        gs[1], dtype=np.float32)

        x_first = np.full(gs[0], x[0])
        x_last = np.full(gs[0], x[-1])
        y_first = np.full(gs[1], y[0])
        y_last = np.full(gs[1], y[-1])
        border = np.concatenate((np.column_stack((x_first, x)), np.column_stack(
            (x_last, x)), np.column_stack((y, y_first))[1:-1], np.column_stack((y, y_last))[1:-1]))
        return border
    
    def real_pairs(self, grid):
        pass
    
    def prediction_pairs(self, grid):
        pass
    
    def plot_area(self, grid):
        self.x = np.linspace(limits[0][0], limits[0][1], grid_size[0])
        self.y = np.linspace(limits[1][0], limits[1][1], grid_size[1])
        self.x, self.y = np.meshgrid(self.x, self.y)
        pred_coord = []
        for _x in self.x[0]:
            for _y in self.x[0]:
                pred_coord.append([_x, _y])
        pred_coord = np.array(pred_coord)
        self.true_u = real_u((self.x, self.y))
        self.pred_u = model.predict(pred_coord).ravel().reshape(grid_size)
        self.x_limits = limits[0]
        self.y_limits = limits[1]

    @staticmethod
    def __mesh_to_pairs(meshgrid: list[np.ndarray]) -> npt.NDArray:
        return np.column_stack((meshgrid[0].flatten(), meshgrid[1].flatten()))
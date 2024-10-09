from dataset import get_area, get_border
from accuracy import AccuracyCalc
import numpy as np
from plotting import NNPlots


class ModelInfo:

    def __init__(self, model, real_u, limits, grid_size):
        self.grid_size = grid_size
        self.real_u = real_u
        self.model = model
        self.limits = limits
        self.inner_area = get_area(grid_size, limits[0], limits[1])
        self.border = get_border(grid_size, limits[0], limits[1])
        self.area = np.concatenate((self.inner_area, self.border), axis=0)
        self.error_calc = AccuracyCalc(model, real_u, self.area)
        self.border_error_calc = AccuracyCalc(model, real_u, self.border)
        self.inner_error_calc = AccuracyCalc(model, real_u, self.inner_area)
        self.plotter = NNPlots(limits, grid_size, model, real_u)

    def change_plotter(self, limits=None, grid_size=None, model=None, real_u=None):
        if limits == None:
            limits = self.limits
        if grid_size == None:
            grid_size = self.grid_size
        if model == None:
            model = self.model
        if real_u == None:
            real_u = self.real_u
        self.plotter = NNPlots(limits, grid_size, model, real_u)

import numpy as np
from data_gen import DataGenerator
# from typing 


class AccuracyCalc:
    EPSILON = 1e-10

    """
    Area isn't passed as an argument in this class,
    it's like a state of an object
    """

    # TODO: divide area into border and inside
    def __init__(self, dg: DataGenerator, grid):
        # dg.prediction_pairs(grid)
        # dg.real_pairs
        # self._real_val = 
        # self._pred_val = 
        pass

    def update_grid(self, grid):
        self._re = 
        

    def _to_percent(error_function):
        def wrapper_to_percent(*args, **kwargs):
            return error_function(*args, **kwargs) * 100

        return wrapper_to_percent

    @_to_percent
    def good_perc_rel(self, rel_dis: float) -> float:
        """
        Percent of close values, by relative distance
        """
        return np.sum(
            np.isclose(self._pinn_val, self._actual_val, rtol=rel_dis)
        ) / len(self._area)

    @_to_percent
    def good_perc_abs(self, abs_dis: float) -> float:
        """
        Percent of close values, by absolute distance
        """
        return np.sum(
            np.isclose(self._pinn_val, self._actual_val, atol=abs_dis)
        ) / len(self._area)

    @_to_percent
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5570302/
    def ve_acc(self) -> float:
        """
        Variance Explained Accuracy
        """
        mean = np.mean(self._actual_val)
        return 1 - np.sum(np.square(self._actual_val - self._pinn_val)) / np.sum(
            np.square(self._actual_val - mean)
        )

    def maape(self):
        """
        Mean Arctangent Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return np.mean(
            np.arctan(
                np.abs(
                    (self._actual_val - self._pinn_val)
                    / (self._actual_val + AccuracyCalc.EPSILON)
                )
            )
        )

    def mse(self):
        """
        Mean Squared Error
        """
        return np.mean(np.square(self._actual_val - self._pinn_val))

    def maxe(self):
        """
        Maximum Error
        """
        return np.max(np.abs(self._actual_val - self._pinn_val))

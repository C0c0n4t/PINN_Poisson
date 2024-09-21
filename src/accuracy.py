import numpy as np


class AccuracyCalc:
    EPSILON = 1e-10

    def __init__(self, model, real_function, dataset):
        self._model = model
        self._real_function = real_function
        self._dataset = dataset
        self._area = None

    @property
    def area(self, area: np.array):  # like getter
        return self._area

    @area.setter
    def area(self, area: np.array):  # like setter
        self._area = area

    def _to_percent(error_function):
        def wrapper_to_percent(*args, **kwargs):
            return error_function(*args, **kwargs) * 100

        return wrapper_to_percent

    @_to_percent
    def good_perc_rel(self, area: np.ndarray | None, rel_dis: float):
        """
        Percent of close values, by relative distance
        """
        if area.isinstance(None):
            return np.sum(
                np.isclose(
                    self._model(self._area),
                    self._real_function(self._area),
                    rtol=rel_dis,
                )
            ) / len(self._area)
        else:
            return np.sum(
                np.isclose(self._model(area), self._real_function(area), rtol=rel_dis)
            ) / len(area)

    # @_to_percent
    # def good_perc_abs(actual: np.ndarray, predicted: np.ndarray):
    #     """
    #     Percent of close values, by absolute distance
    #     """
    #     return np.sum(np.isclose(predicted, actual, atol=0.1)) / len(total_predictions)

    # @_to_percent
    # # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5570302/
    # def ve_acc(actual: np.ndarray, predicted: np.ndarray):
    #     """
    #     Variance Explained Accuracy
    #     """
    #     return 1 - np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual)))

    # def maape(actual: np.ndarray, predicted: np.ndarray):
    #     """
    #     Mean Arctangent Absolute Percentage Error
    #     Note: result is NOT multiplied by 100
    #     """
    #     return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))

    def mse(actual: np.ndarray, predicted: np.ndarray):
        """
        Mean Squared Error
        """
        return np.mean(np.square(actual - predicted))

    # def maxe()

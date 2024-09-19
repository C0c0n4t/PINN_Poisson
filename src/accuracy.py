import numpy as np

EPSILON = 1e-10


#TODO: maybe make a class, that gets model and real function and compare them


def _to_percent(error_function):
    def wrapper_to_percent(*args, **kwargs):
        return error_function(*args, **kwargs) * 100
    return wrapper_to_percent


@_to_percent
def good_perc_rel(actual: np.ndarray, predicted: np.ndarray):
    """
    Percent of close values, by relative distance
    """
    return np.sum(np.isclose(predicted, actual, rtol=0.1)) / len(total_predictions)


@_to_percent
def good_perc_abs(actual: np.ndarray, predicted: np.ndarray):
    """
    Percent of close values, by absolute distance
    """
    return np.sum(np.isclose(predicted, actual, atol=0.1)) / len(total_predictions)


@_to_percent
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5570302/
def ve_acc(actual: np.ndarray, predicted: np.ndarray):
    """
    Variance Explained Accuracy
    """
    return 1 - np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual)))


def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Squared Error
    """
    return np.mean(np.square(actual - predicted))

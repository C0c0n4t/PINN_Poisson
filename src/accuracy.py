import numpy as np

EPSILON = 1e-10


def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def good_perc(actual: np.ndarray, predicted: np.ndarray):
    # Calculate the number of correct predictions
    correct_predictions = np.sum(np.isclose(predicted, actual, rtol=0.1))

    # Calculate the total number of predictions
    total_predictions = len(predicted)

    return correct_predictions / total_predictions

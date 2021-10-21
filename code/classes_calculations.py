import numpy as np

def classes_calculations(input):
    """calculates the classes for a given input

    Args:
        input ([Number]): the input signal

    Returns:
        (Number): the two classes
    """
    counts, _ = np.histogram(input, bins = int(input.max() + 1), range = (0, int(input.max())))
    return np.nonzero(counts)[0]
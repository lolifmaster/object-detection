from typing import Sequence
from cv1 import tools
import numpy as np


def mean(src, kernel_size: Sequence[int]):
    """
    Apply mean filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (Sequence[int]): The kernel size.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size should be an Sequence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size should be an Sequence of length 2")

    kernel = np.ones(kernel_size) / (kernel_size[0] * kernel_size[1])
    return tools.filter_2d(src, kernel)

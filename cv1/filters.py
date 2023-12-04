from cv1 import tools
import numpy as np


def mean(src, kernel_size):
    """
    Apply mean filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (int): The kernel size.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not tools.is_gray_scale(src):
        raise ValueError("src should be a gray scale image")

    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return tools.filter_2d(src, kernel)

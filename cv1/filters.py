from typing import Sequence
from cv1 import tools
import numpy as np
from cv1.tools import range


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


def median(src, kernel_size: Sequence[int]):
    """
    Apply median filter to the source image.

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

    dst = np.zeros_like(src)
    pad_width = ((kernel_size[0] - 1) // 2, kernel_size[0] // 2), ((kernel_size[1] - 1) // 2, kernel_size[1] // 2)
    padded_src = np.pad(src, pad_width, mode='edge')

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = np.median(padded_src[i:i + kernel_size[0], j:j + kernel_size[1]])

    return dst


def gaussian(src, kernel_size: Sequence[int], sigma: float):
    """
    Apply gaussian filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (Sequence[int]): The kernel size.
        sigma (float): The standard deviation of the gaussian distribution.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size should be an Sequence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size should be an Sequence of length 2")
    if sigma <= 0:
        raise ValueError("sigma should be greater than 0")

    kernel = np.zeros(kernel_size)
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            kernel[i, j] = np.exp(-((i - kernel_size[0] // 2) ** 2 + (j - kernel_size[1] // 2) ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)

    return tools.filter_2d(src, kernel)

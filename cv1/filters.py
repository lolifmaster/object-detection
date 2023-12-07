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


def median(src, kernel_size: Sequence[int], mode='edge', constant_values=0):
    """
    Apply median filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (Sequence[int]): The kernel size.
        mode (str): The padding mode. Can be 'constant', 'edge'.
        constant_values (int): The constant value to use if mode='constant'.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not tools.is_gray_scale(mode):
        return

    dst = np.zeros_like(src)
    pad_width = ((kernel_size[0] // 2, kernel_size[0] // 2), (kernel_size[1] // 2, kernel_size[1] // 2))
    padded_src = tools.pad(src, pad_width, mode, constant_values)

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
    if sigma <= 0:
        sigma = src.std()

    kernel = np.zeros(kernel_size)
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            kernel[i, j] = np.exp(-((i - kernel_size[0] // 2) ** 2 + (j - kernel_size[1] // 2) ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)

    return tools.filter_2d(src, kernel)


def laplacian(src):
    """
    Apply laplacian filter to the source image.

    Args:
        src (numpy.ndarray): The source image.

    Returns:
        numpy.ndarray: The filtered image.
    """
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return tools.filter_2d(src, kernel, clip=False)


def edge_detection(src):
    """
    Apply edge detection filter to the source image.

    Args:
        src (numpy.ndarray): The source image.

    Returns:
        numpy.ndarray: The filtered image.
    """
    kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    return tools.filter_2d(src, kernel)


def sharpen(src):
    """
    Apply sharpen filter to the source image.

    Args:
        src (numpy.ndarray): The source image.

    Returns:
        numpy.ndarray: The filtered image.
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen_img = tools.filter_2d(src, kernel, mode='constant', constant_values=0)
    return sharpen_img


def emboss(src):
    """
    Apply emboss filter to the source image.

    Args:
        src (numpy.ndarray): The source image.

    Returns:
        numpy.ndarray: The filtered image.
    """
    kernel_left = np.array(
        [[0, -1, -1],
         [1, 0, -1],
         [1, 1, 0]])
    kernel_right = np.array(
        [[-1, -1, 0],
         [-1, 0, 1],
         [0, 1, 1]])

    left = tools.filter_2d(src, kernel_left)
    right = tools.filter_2d(src, kernel_right)
    combined = np.zeros_like(src)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            combined[i, j] = max(left[i, j], right[i, j])
    return combined


def bilateral(src, kernel_size: Sequence[int], sigma_s: float, sigma_r: float):
    """
    Apply bilateral filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (Sequence[int]): The kernel size.
        sigma_s (float): The standard deviation of the spatial gaussian distribution.
        sigma_r (float): The standard deviation of the range gaussian distribution.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not tools.is_gray_scale(src):
        return

    if sigma_s <= 0:
        sigma_s = src.std()
    if sigma_r <= 0:
        sigma_r = src.std()

    dst = np.zeros_like(src)
    pad_width = ((kernel_size[0] // 2, kernel_size[0] // 2), (kernel_size[1] // 2, kernel_size[1] // 2))
    padded_src = tools.pad(src, pad_width, 'edge')

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            kernel = np.zeros(kernel_size)
            for k in range(kernel_size[0]):
                for l in range(kernel_size[1]):
                    kernel[k, l] = np.exp(-((k - kernel_size[0] // 2) ** 2 + (l - kernel_size[1] // 2) ** 2) / (
                            2 * sigma_s ** 2)) * np.exp(
                        -((padded_src[i, j] - padded_src[i + k, j + l]) ** 2) / (2 * sigma_r ** 2))
            kernel /= np.sum(kernel)
            dst[i, j] = np.sum(kernel * padded_src[i:i + kernel_size[0], j:j + kernel_size[1]])

    return dst

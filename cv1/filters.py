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
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size should be an Sequence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size should be an Sequence of length 2")

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
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size should be an Sequence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size should be an Sequence of length 2")
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


def erode(src, kernel_size: Sequence[int], *, iterations: int = 1):
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size should be an Sequence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size should be an Sequence of length 2")
        
    dst = np.zeros_like(src)
    pad_height = (kernel_size[0] - 1) // 2
    pad_width = (kernel_size[1] - 1) // 2
    temp_img = np.copy(src) 

    for _ in range(iterations):

        padded_dst = tools.pad(temp_img, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

        for i in range(temp_img.shape[0]):
            for j in range(temp_img.shape[1]):
                dst[i, j] = np.min(padded_dst[i:i + kernel_size[0], j:j + kernel_size[1]])
        
        np.copyto(temp_img, dst)

    return dst

def dilate(src, kernel_size: Sequence[int], *, iterations: int = 1 ):
    """
    Apply dillatatation to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (Sequence[int]): The kernel size.

    Raises:
        numpy.ndarray: The filtered image.
    """
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size should be an Sequence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size should be an Sequence of length 2")
    
    dst = np.zeros_like(src)
    pad_width = ((kernel_size[0] - 1) // 2, kernel_size[0] // 2), ((kernel_size[1] - 1) // 2, kernel_size[1] // 2)
    temp_img= src
    
    for _ in range(iterations):
        padded_dst = tools.pad(temp_img, pad_width, mode='edge')
        for i in range(temp_img.shape[0]):
            for j in range(temp_img.shape[1]):
                dst[i, j] = np.max(padded_dst[i:i + kernel_size[0], j:j + kernel_size[1]])
        
        temp_img = dst

    return dst
import numpy as np


def range(stop, start=0, step=1):
    current = start
    while current < stop if step > 0 else current > stop:
        yield current 
        current += step


def filter_2d(src, kernel):
    """
    Apply a 2D filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel (numpy.ndarray): The filter kernel.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not isinstance(src, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise ValueError("src and kernel should be numpy arrays")
    if len(src.shape) < 2 or len(kernel.shape) < 2:
        raise ValueError("src and kernel should be 2D arrays")

    dst = np.zeros_like(src)
    pad_width = ((kernel.shape[0] - 1) // 2, kernel.shape[0] // 2), ((kernel.shape[1] - 1) // 2, kernel.shape[1] // 2)
    padded_src = np.pad(src, pad_width, mode='edge')

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = round(np.sum(padded_src[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel))

    return dst


def is_gray_scale(src):
    """
    Check if the source image is gray scale.

    Args:
        src (numpy.ndarray): The source image.

    Returns:
        bool: True if the image is gray scale, False otherwise.
    """
    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    return len(src.shape) == 2
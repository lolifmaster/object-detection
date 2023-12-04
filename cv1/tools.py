import numpy as np


def filter_2d(src, kernel, *, ddepth=-1):
    """
    Apply a 2D filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        ddepth (int): The desired depth of the destination image.
        kernel (numpy.ndarray): The filter kernel.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not isinstance(src, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise ValueError("src and kernel should be numpy arrays")
    if len(src.shape) < 2 or len(kernel.shape) < 2:
        raise ValueError("src and kernel should be 2D arrays")

    dst = np.zeros_like(src)
    pad_size = kernel.shape[0] // 2
    src_padded = np.pad(src, pad_size, mode='constant')

    for index, _ in np.ndenumerate(src):
        i, j = index
        dst[i, j] = np.sum(src_padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

    if ddepth != -1:
        dst = dst.astype(ddepth)

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

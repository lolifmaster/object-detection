import numpy as np


def range(stop, start=0, step=1):
    current = start
    while current < stop if step > 0 else current > stop:
        yield current
        current += step


def pad(src, pad_width, mode='constant', constant_values=0):
    """
    Pad an image.

    Args:
        src (numpy.ndarray): The source image.
        pad_width (Sequence[Sequence[int]]): The padding width.
        mode (str): The padding mode. Can be 'constant', 'edge'.
        constant_values (int): The constant value to use if mode='constant'.

    Returns:
        numpy.ndarray: The padded image.
    """
    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    padded_src = np.zeros(
        (src.shape[0] + pad_width[0][0] + pad_width[0][1], src.shape[1] + pad_width[1][0] + pad_width[1][1]))

    if mode == 'constant':
        padded_src[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]] = src
        padded_src[:pad_width[0][0], :] = constant_values
        padded_src[-pad_width[0][1]:, :] = constant_values
        padded_src[:, :pad_width[1][0]] = constant_values
        padded_src[:, -pad_width[1][1]:] = constant_values

    elif mode == 'edge':
        padded_src[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]] = src
        padded_src[:pad_width[0][0], pad_width[1][0]:-pad_width[1][1]] = src[0]
        padded_src[-pad_width[0][1]:, pad_width[1][0]:-pad_width[1][1]] = src[-1]
        padded_src[pad_width[0][0]:-pad_width[0][1], :pad_width[1][0]] = src[:, 0].reshape(-1, 1)
        padded_src[pad_width[0][0]:-pad_width[0][1], -pad_width[1][1]:] = src[:, -1].reshape(-1, 1)
        padded_src[:pad_width[0][0], :pad_width[1][0]] = src[0, 0]
        padded_src[:pad_width[0][0], -pad_width[1][1]:] = src[0, -1]
        padded_src[-pad_width[0][1]:, :pad_width[1][0]] = src[-1, 0]
        padded_src[-pad_width[0][1]:, -pad_width[1][1]:] = src[-1, -1]

    return padded_src


def filter_2d(src, kernel, mode='edge', constant_values=0):
    """
    Apply a 2D filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel (numpy.ndarray): The filter kernel.
        mode (str): The padding mode. Can be 'constant', 'edge'.
        constant_values (int): The constant value to use if mode='constant'.

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not isinstance(src, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise ValueError("src and kernel should be numpy arrays")
    if len(src.shape) < 2 or len(kernel.shape) < 2:
        raise ValueError("src and kernel should be 2D arrays")

    dst = np.zeros_like(src)
    pad_width = ((kernel.shape[0] - 1) // 2, kernel.shape[0] // 2), ((kernel.shape[1] - 1) // 2, kernel.shape[1] // 2)
    padded_src = pad(src, pad_width, mode=mode, constant_values=constant_values)

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

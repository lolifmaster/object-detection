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


def sum_array(array: np.array):
    """
    Sum all the elements in the array.

    Args:
        array (numpy.ndarray): The array.

    Returns:
        float: The sum of all the elements in the array.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("array should be a numpy array")

    array_sum = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array_sum += array[i, j]

    return array_sum


def median(array: np.array):
    """
    Median all the elements in the array.

    Args:
        array (numpy.ndarray): The array.

    Returns:
        float: The median of all the elements in the array.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("array should be a numpy array")

    array = array.flatten()
    array = sorted(array)
    return array[len(array) // 2]


def clip_array(array, min_value, max_value):
    """
    Clip the value between min_value and max_value.

    Args:
        array (np.array): The array.
        min_value (float): The minimum value.
        max_value (float): The maximum value.

    Returns:
        float: The clipped value.
    """
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] < min_value:
                array[i, j] = min_value
            elif array[i, j] > max_value:
                array[i, j] = max_value

    return array


def filter_2d(src, kernel, mode='edge', constant_values=0, clip=True):
    """
    Apply a 2D filter to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel (numpy.ndarray): The filter kernel.
        mode (str): The padding mode. Can be 'constant', 'edge'.
        constant_values (int): The constant value to use if mode='constant'.
        clip (bool): Whether to clip the output image to [0, 255].

    Returns:
        numpy.ndarray: The filtered image.
    """
    if not is_gray_scale(src):
        raise ValueError("src should be a gray scale image")

    rows, cols = src.shape
    k_rows, k_cols = kernel.shape

    # Calculate the padding needed to ensure the output size is the same as the input size
    pad_rows = k_rows // 2
    pad_cols = k_cols // 2

    # Pad the image
    padded_image = pad(src, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode, constant_values)

    # Initialize the output image
    output_image = np.zeros_like(src, dtype=np.float32)

    # Convolution operation
    for i in range(rows):
        for j in range(cols):
            output_image[i, j] = sum_array(padded_image[i:i + k_rows, j:j + k_cols] * kernel)

    if clip:
        output_image = clip_array(output_image, 0, 255)

    return output_image.astype(np.uint8)


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

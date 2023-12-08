import numpy as np
from cv1.shapes import Shape
from typing import Sequence


def range(stop, start=0, step=1):
    current = start
    while current < stop if step > 0 else current > stop:
        yield current
        current += step


def pad(src, pad_width, mode="constant", constant_values=0):
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
    if not isinstance(pad_width, Sequence):
        raise ValueError("pad_width should be an Sequence")

    if len(pad_width) != 2:
        raise ValueError("pad_width should be an Sequence of length 2")

    if mode not in ["constant", "edge"]:
        raise ValueError("mode should be 'constant' or 'edge'")

    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    if len(src.shape) != 2:
        raise ValueError("src should be a 2D array")

    padded_src = np.zeros(
        (
            src.shape[0] + pad_width[0][0] + pad_width[0][1],
            src.shape[1] + pad_width[1][0] + pad_width[1][1],
        )
    )

    if mode == "constant":
        padded_src[
            pad_width[0][0] : -pad_width[0][1], pad_width[1][0] : -pad_width[1][1]
        ] = src
        padded_src[: pad_width[0][0], :] = constant_values
        padded_src[-pad_width[0][1] :, :] = constant_values
        padded_src[:, : pad_width[1][0]] = constant_values
        padded_src[:, -pad_width[1][1] :] = constant_values

    elif mode == "edge":
        padded_src[
            pad_width[0][0] : -pad_width[0][1], pad_width[1][0] : -pad_width[1][1]
        ] = src
        padded_src[: pad_width[0][0], pad_width[1][0] : -pad_width[1][1]] = src[0]
        padded_src[-pad_width[0][1] :, pad_width[1][0] : -pad_width[1][1]] = src[-1]
        padded_src[pad_width[0][0] : -pad_width[0][1], : pad_width[1][0]] = src[
            :, 0
        ].reshape(-1, 1)
        padded_src[pad_width[0][0] : -pad_width[0][1], -pad_width[1][1] :] = src[
            :, -1
        ].reshape(-1, 1)
        padded_src[: pad_width[0][0], : pad_width[1][0]] = src[0, 0]
        padded_src[: pad_width[0][0], -pad_width[1][1] :] = src[0, -1]
        padded_src[-pad_width[0][1] :, : pad_width[1][0]] = src[-1, 0]
        padded_src[-pad_width[0][1] :, -pad_width[1][1] :] = src[-1, -1]

    return padded_src


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


def filter_2d(src, kernel, mode="edge", constant_values=0, clip=True):
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
    if not isinstance(kernel, np.ndarray):
        raise ValueError("kernel should be a numpy array")

    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    if len(kernel.shape) != 2:
        raise ValueError("kernel should be a 2D array")

    if len(src.shape) != 2:
        raise ValueError("src should be a 2D array")

    rows, cols = src.shape
    k_rows, k_cols = kernel.shape

    # Calculate the padding needed to ensure the output size is the same as the input size
    pad_rows = k_rows // 2
    pad_cols = k_cols // 2

    # Pad the image
    padded_image = pad(
        src, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode, constant_values
    )

    # Initialize the output image
    output_image = np.zeros_like(src, dtype=np.float32)

    # Convolution operation
    for i in range(rows):
        for j in range(cols):
            output_image[i, j] = np.sum(
                padded_image[i : i + k_rows, j : j + k_cols] * kernel
            )

    if clip:
        output_image = clip_array(output_image, 0, 255)

    return output_image.astype(np.uint8)


def bgr2hsv(src):
    """
    Convert a BGR image to HSV.

    Args:
        src (numpy.ndarray): The source image.

    Returns:
        numpy.ndarray: The HSV image.
    """
    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    if src.shape[2] != 3:
        raise ValueError("src should be a BGR image")
    np.seterr(divide="ignore", invalid="ignore")

    hsv = np.zeros_like(src)

    # Extract the BGR channels and normalize
    b, g, r = src[:, :, 0] / 255.0, src[:, :, 1] / 255.0, src[:, :, 2] / 255.0

    # Calculate the value channel
    v = np.max(src, axis=2) / 255.0
    m = np.min(src, axis=2) / 255.0

    # Calculate the saturation channel
    s = np.zeros_like(v)
    non_zero_v = v != 0
    s[non_zero_v] = (v[non_zero_v] - m[non_zero_v]) / v[non_zero_v] * 255

    # Calculate the hue channel
    h = np.zeros_like(v)

    delta = v - m

    h[v == r] = 60 * (g[v == r] - b[v == r]) / (delta[v == r])
    h[v == g] = 120 + 60 * (b[v == g] - r[v == g]) / (delta[v == g])
    h[v == b] = 240 + 60 * (r[v == b] - g[v == b]) / (delta[v == b])
    h[v == 0] = 0

    # Convert the hue channel to degrees
    h[h < 0] += 360

    # Normalize the hue channel
    h = (h / 360.0) * 179.0
    # Merge the channels
    hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = h, s, np.max(src, axis=2)

    return hsv


def create_shape(shape_type: Shape, size: int):
    """
    Create a shape.

    Args:
        shape_type (Shape): The shape type.
        size (int): The size of the shape.

    Returns:
        numpy.ndarray: The shape.
    """

    match shape_type:
        case Shape.SQUARE:
            return np.ones((size, size), dtype=np.uint8)
        case Shape.RECT:
            return np.array(
                [
                    [
                        0
                        if (i - size // 2) ** 2 + (j - size // 2) ** 2
                        > (size // 2) ** 2
                        else 1
                        for j in range(size)
                    ]
                    for i in range(size)
                ],
                dtype=np.uint8,
            )
        case Shape.CROSS:
            return np.array(
                [
                    [1 if i == size // 2 or j == size // 2 else 0 for j in range(size)]
                    for i in range(size)
                ],
                dtype=np.uint8,
            )
        case Shape.TRIANGLE:
            return np.array(
                [[1 if i >= j else 0 for j in range(size)] for i in range(size)],
                dtype=np.uint8,
            )
        case Shape.DIAMOND:
            return np.array(
                [
                    [
                        1 if abs(i - size // 2) + abs(j - size // 2) <= size // 2 else 0
                        for j in range(size)
                    ]
                    for i in range(size)
                ],
                dtype=np.uint8,
            )
        case _:
            raise ValueError("Invalid shape type")


def bitwise_and(src: np.array, mask):
    result = np.zeros_like(src)
    result[mask == 255] = src[mask == 255]

    return result


def in_range(src: np.array, lower_bound, upper_bound):
    """
    Check if the pixels in the source image are within the specified range.

    Args:
        src (numpy.ndarray): The source image.
        lower_bound (Sequence[int]): The lower bound.
        upper_bound (Sequence[int]): The upper bound.

    Returns:
        numpy.ndarray: The mask.
    """
    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    if len(src.shape) != 3:
        raise ValueError("src should be a 3D array")

    if len(lower_bound) != 3:
        raise ValueError("lower_bound should be a Sequence of length 3")

    if len(upper_bound) != 3:
        raise ValueError("upper_bound should be a Sequence of length 3")

    if not isinstance(lower_bound, Sequence):
        raise ValueError("lower_bound should be a Sequence")

    if not isinstance(upper_bound, Sequence):
        raise ValueError("upper_bound should be a Sequence")

    mask = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if (
                lower_bound[0] <= src[i, j, 0] <= upper_bound[0]
                and lower_bound[1] <= src[i, j, 1] <= upper_bound[1]
                and lower_bound[2] <= src[i, j, 2] <= upper_bound[2]
            ):
                mask[i, j] = 255

    return mask


def threshold(src: np.array, threshold_value: int, max_value: int):
    """
    Apply thresholding to the source image.

    Args:
        src (numpy.ndarray): The source image.
        threshold_value (int): The threshold value.
        max_value (int): The maximum value.

    Returns:
        numpy.ndarray: The threshold image.
    """
    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    if len(src.shape) != 2:
        raise ValueError("src should be a 2D array")

    dst = np.zeros_like(src, dtype=np.uint8)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = max_value if src[i, j] > threshold_value else 0

    return dst

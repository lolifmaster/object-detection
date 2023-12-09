from typing import Sequence, Literal
from cv1 import tools
import numpy as np
from cv1.shapes import Shape
import cv2


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
    pad_width = (
        (kernel_size[0] // 2, kernel_size[0] // 2),
        (kernel_size[1] // 2, kernel_size[1] // 2),
    )
    padded_src = tools.pad(src, pad_width, "edge")

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = np.median(
                padded_src[i : i + kernel_size[0], j : j + kernel_size[1]]
            )

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
            kernel[i, j] = np.exp(
                -((i - kernel_size[0] // 2) ** 2 + (j - kernel_size[1] // 2) ** 2)
                / (2 * sigma**2)
            )
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
    sharpen_img = tools.filter_2d(src, kernel, mode="constant", constant_values=0)
    return sharpen_img


def emboss(src):
    """
    Apply emboss filter to the source image.

    Args:
        src (numpy.ndarray): The source image.

    Returns:
        numpy.ndarray: The filtered image.
    """
    emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

    emboss_img = tools.filter_2d(src, emboss_kernel, mode="edge")
    return emboss_img


def sobel(image, direction: Literal["x", "y", "xy"] = "xy"):
    """
    Apply sobel filter to the source image.

    Args:
        image (numpy.ndarray): The source image.
        direction (Literal["x", "y", "xy"]): The direction of the sobel filter.
    :return:
        numpy.ndarray: The filtered image.
    """

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    if direction == "x":
        sobel_kernel = kernel_x
    elif direction == "y":
        sobel_kernel = kernel_y
    elif direction == "xy":
        sobel_kernel = kernel_x + kernel_y
    else:
        raise ValueError("direction should be x, y or xy")

    sobel_img = tools.filter_2d(image, sobel_kernel, mode="edge")

    return sobel_img


def erode(
    src, kernel_size: int, *, iterations: int = 1, kernel_shape: Shape = Shape.RECT
):
    """
    Apply erosion to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (int): The kernel size.
        iterations (int): The number of iterations.
        kernel_shape (Shape): The kernel shape.

    Returns:
        numpy.ndarray: The filtered image.
    """
    dst = np.zeros_like(src)
    pad_width = ((kernel_size - 1) // 2, kernel_size // 2), (
        (kernel_size - 1) // 2,
        kernel_size // 2,
    )
    temp_img = src.copy()

    for _ in range(iterations):
        padded_dst = tools.pad(temp_img, pad_width, mode="edge")
        for i in range(temp_img.shape[0]):
            for j in range(temp_img.shape[1]):
                match kernel_shape:
                    case Shape.RECT:
                        dst[i, j] = np.min(
                            padded_dst[i : i + kernel_size, j : j + kernel_size]
                        )
                    case Shape.CROSS:
                        dst[i, j] = np.min(
                            [
                                padded_dst[i : i + kernel_size, j + kernel_size // 2],
                                padded_dst[i + kernel_size // 2, j : j + kernel_size],
                            ]
                        )
                    case _:
                        raise ValueError("kernel_shape should be RECT or CROSS")

        temp_img = dst

    return dst


def dilate(
    src, kernel_size: int, *, iterations: int = 1, kernel_shape: Shape = Shape.RECT
):
    """
    Apply dilatation to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (int): The kernel size.
        iterations (int): The number of iterations.
        kernel_shape (Shape): The kernel shape.

    Raises:
        numpy.ndarray: The filtered image.
    """
    dst = np.zeros_like(src)
    pad_width = ((kernel_size - 1) // 2, kernel_size // 2), (
        (kernel_size - 1) // 2,
        kernel_size // 2,
    )
    temp_img = src

    for _ in range(iterations):
        padded_dst = tools.pad(temp_img, pad_width, mode="edge")
        for i in range(temp_img.shape[0]):
            for j in range(temp_img.shape[1]):
                match kernel_shape:
                    case Shape.RECT:
                        dst[i, j] = np.max(
                            padded_dst[i : i + kernel_size, j : j + kernel_size]
                        )
                    case Shape.CROSS:
                        dst[i, j] = np.max(
                            [
                                padded_dst[i : i + kernel_size, j + kernel_size // 2],
                                padded_dst[i + kernel_size // 2, j : j + kernel_size],
                            ]
                        )
                    case _:
                        raise ValueError("kernel_shape should be RECT or CROSS")

        temp_img = dst

    return dst


def opening(
    src, kernel_size: int, *, iterations: int = 1, kernel_shape: Shape = Shape.RECT
):
    """
    Apply opening to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (int): The kernel size.
        iterations (int): The number of iterations.
        kernel_shape (Shape): The kernel shape.

    Returns:
        numpy.ndarray: The filtered image.
    """
    return dilate(
        erode(src, kernel_size, iterations=iterations, kernel_shape=kernel_shape),
        kernel_size,
        iterations=iterations,
        kernel_shape=kernel_shape,
    )


def closing(
    src, kernel_size: int, *, iterations: int = 1, kernel_shape: Shape = Shape.RECT
):
    """
    Apply closing to the source image.

    Args:
        src (numpy.ndarray): The source image.
        kernel_size (int): The kernel size.
        iterations (int): The number of iterations.
        kernel_shape (Shape): The kernel shape.

    Returns:
        numpy.ndarray: The filtered image.
    """
    return erode(
        dilate(src, kernel_size, iterations=iterations, kernel_shape=kernel_shape),
        kernel_size,
        iterations=iterations,
        kernel_shape=kernel_shape,
    )

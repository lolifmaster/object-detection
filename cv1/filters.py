from typing import Sequence, Literal
from cv1 import tools
import numpy as np
from cv1.shapes import Shape


def mean(src, kernel_size: Sequence[int]):
    """
    Applique un filtre moyenne à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        kernel_size (Sequence[int]): La taille du noyau.

    Returns:
        numpy.ndarray: L'image filtrée.
    """
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size devrait être une séquence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size devrait être une séquence de longueur 2")

    kernel = np.ones(kernel_size) / (kernel_size[0] * kernel_size[1])
    return tools.filter_2d(src, kernel)


def median(src, kernel_size: Sequence[int]):
    """
    Applique un filtre médian à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        kernel_size (Sequence[int]): La taille du noyau.

    Returns:
        numpy.ndarray: L'image filtrée.
    """
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size devrait être une séquence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size devrait être une séquence de longueur 2")

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
    Applique un filtre gaussien à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        kernel_size (Sequence[int]): La taille du noyau.
        sigma (float): L'écart-type de la distribution gaussienne.

    Returns:
        numpy.ndarray: L'image filtrée.
    """
    if not isinstance(kernel_size, Sequence):
        raise ValueError("kernel_size devrait être une séquence")
    if len(kernel_size) != 2:
        raise ValueError("kernel_size devrait être une séquence de longueur 2")
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
    Applique un filtre laplacien à l'image source.

    Args:
        src (numpy.ndarray): L'image source.

    Returns:
        numpy.ndarray: L'image filtrée.
    """
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return tools.filter_2d(src, kernel, clip=False)


def edge_detection(src):
    """
    Applique un filtre de détection de contours à l'image source.

    Args:
        src (numpy.ndarray): L'image source.

    Returns:
        numpy.ndarray: L'image filtrée.
    """
    kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    return tools.filter_2d(src, kernel)


def sharpen(src):
    """
    Applique un filtre de netteté à l'image source.

    Args:
        src (numpy.ndarray): L'image source.

    Returns:
        numpy.ndarray: L'image filtrée.
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen_img = tools.filter_2d(src, kernel, mode="constant", constant_values=0)
    return sharpen_img


def emboss(src):
    """
    Applique un filtre en relief à l'image source.

    Args:
        src (numpy.ndarray): L'image source.

    Returns:
        numpy.ndarray: L'image filtrée.
    """
    emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

    emboss_img = tools.filter_2d(src, emboss_kernel, mode="edge")
    return emboss_img


def sobel(image, direction: Literal["x", "y", "xy"] = "xy"):
    """
    Applique un filtre de Sobel à l'image source.

    Args:
        image (numpy.ndarray): L'image source.
        direction (Literal["x", "y", "xy"]): La direction du filtre de Sobel.

    Returns:
        numpy.ndarray: L'image filtrée.
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
        raise ValueError("direction devrait être x, y ou xy")

    sobel_img = tools.filter_2d(image, sobel_kernel, mode="edge")

    return sobel_img


def erode(
    src, kernel_size: int, *, iterations: int = 1, kernel_shape: Shape = Shape.RECT
):
    """
    Applique une érosion à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        kernel_size (int): La taille du noyau.
        iterations (int): Le nombre d'itérations.
        kernel_shape (Shape): La forme du noyau.

    Returns:
        numpy.ndarray: L'image filtrée.
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
                        raise ValueError("kernel_shape devrait être RECT ou CROSS")

        temp_img = dst

    return dst


def dilate(
    src, kernel_size: int, *, iterations: int = 1, kernel_shape: Shape = Shape.RECT
):
    """
    Applique une dilatation à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        kernel_size (int): La taille du noyau.
        iterations (int): Le nombre d'itérations.
        kernel_shape (Shape): La forme du noyau.

    Returns:
        numpy.ndarray: L'image filtrée.
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
                        raise ValueError("kernel_shape devrait être RECT ou CROSS")

        temp_img = dst

    return dst


def opening(
    src, kernel_size: int, *, iterations: int = 1, kernel_shape: Shape = Shape.RECT
):
    """
    Applique une opération d'ouverture à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        kernel_size (int): La taille du noyau.
        iterations (int): Le nombre d'itérations.
        kernel_shape (Shape): La forme du noyau.

    Returns:
        numpy.ndarray: L'image filtrée.
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
    Applique une opération de fermeture à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        kernel_size (int): La taille du noyau.
        iterations (int): Le nombre d'itérations.
        kernel_shape (Shape): La forme du noyau.

    Returns:
        numpy.ndarray: L'image filtrée.
    """
    return erode(
        dilate(src, kernel_size, iterations=iterations, kernel_shape=kernel_shape),
        kernel_size,
        iterations=iterations,
        kernel_shape=kernel_shape,
    )

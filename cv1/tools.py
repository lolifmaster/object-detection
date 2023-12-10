import numpy as np
from cv1.shapes import Shape
from typing import Sequence


# def range(stop, start=0, step=1):
#     current = start
#     while current < stop if step > 0 else current > stop:
#         yield current
#         current += step


def pad(src, pad_width, mode="constant", constant_values=0):
    """
    Rembourre une image.

    Args:
        src (numpy.ndarray): L'image source.
        pad_width (Sequence[Sequence[int]]): La largeur de remplissage.
        mode (str): Le mode de remplissage. Peut être 'constant' ou 'edge'.
        constant_values (int): La valeur constante à utiliser si mode='constant'.

    Returns:
        numpy.ndarray: L'image rembourrée.
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
    Coupe les valeurs entre min_value et max_value.

    Args:
        array (np.array): Le tableau.
        min_value (float): La valeur minimale.
        max_value (float): La valeur maximale.

    Returns:
        float: La valeur coupée.
    """
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] < min_value:
                array[i, j] = min_value
            elif array[i, j] > max_value:
                array[i, j] = max_value

    return array


def filter_2d(
    src, kernel, mode="edge", constant_values=0, clip=True, *, dtype=np.uint8
):
    """
    Applique un filtre 2D à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        kernel (numpy.ndarray): Le noyau du filtre.
        mode (str): Le mode de remplissage. Peut être 'constant' ou 'edge'.
        constant_values (int): La valeur constante à utiliser si mode='constant'.
        clip (bool): Si la sortie doit être coupée à [0, 255].
        dtype (type): Le type de données de l'image de sortie.

    Returns:
        numpy.ndarray: L'image filtrée.
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

    return output_image.astype(dtype)


def bgr2hsv(src):
    """
    Convertit une image BGR en HSV.

    Args:
        src (numpy.ndarray): L'image source.

    Returns:
        numpy.ndarray: L'image HSV.
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
    Crée une forme.

    Args:
        shape_type (Shape): Le type de forme.
        size (int): La taille de la forme.

    Returns:
        numpy.ndarray: La forme.
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
    """
    Effectue l'opération logique ET sur l'image d'entrée avec un masque.

    Args:
        src: L'image source.
        mask: Le masque.

    Returns:
        numpy.ndarray: L'image résultante.
    """
    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    result = np.zeros_like(src)
    result[mask == 255] = src[mask == 255]

    return result


def in_range(src: np.array, lower_bound, upper_bound):
    """
    Vérifie si les pixels de l'image source sont dans la plage spécifiée.

    Args:
        src (numpy.ndarray): L'image source.
        lower_bound (Sequence[int]): La borne inférieure.
        upper_bound (Sequence[int]): La borne supérieure.

    Returns:
        numpy.ndarray: Le masque.
    """
    if not isinstance(src, np.ndarray):
        raise ValueError("src should be a numpy array")

    if len(src.shape) != 3:
        raise ValueError("src should be a 3D array")

    if len(lower_bound) != 3:
        raise ValueError("lower_bound should be a Sequence of length 3")

    if len(upper_bound) != 3:
        raise ValueError("upper_bound should be a Sequence of length 3")

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
    Applique un seuillage à l'image source.

    Args:
        src (numpy.ndarray): L'image source.
        threshold_value (int): La valeur de seuil.
        max_value (int): La valeur maximale.

    Returns:
        numpy.ndarray: L'image seuillée.
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


def bitwise_not(image: np.array):
    """
    Effectue l'opération logique NON sur l'image d'entrée.

    Args:
        image (numpy.ndarray): L'image d'entrée.

    Returns:
        numpy.ndarray: Le résultat de l'opération logique NON.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image should be a numpy array")

    if len(image.shape) != 2:
        raise ValueError("image should be a 2D array")

    return 255 - image


def add_weighted(img1, alpha, img2, beta, gamma):
    """
    Effectue la somme pondérée de deux images.

    Args:
        img1 (numpy.ndarray): La première image d'entrée.
        alpha (float): Poids pour la première image.
        img2 (numpy.ndarray): La deuxième image d'entrée.
        beta (float): Poids pour la deuxième image.
        gamma (float): Scalaire ajouté à chaque somme.

    Returns:
        numpy.ndarray: Le résultat de la somme pondérée.
    """
    if not isinstance(img1, np.ndarray):
        raise ValueError("img1 should be a numpy array")

    if not isinstance(img2, np.ndarray):
        raise ValueError("img2 should be a numpy array")

    if len(img1.shape) != 3:
        raise ValueError("img1 should be a 3D array")

    if len(img2.shape) != 3:
        raise ValueError("img2 should be a 3D array")

    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 should have the same shape")

    result = alpha * img1 + beta * img2 + gamma

    return result

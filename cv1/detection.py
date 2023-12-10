import numpy as np


def dfs(x, y, image, visited, contours):
    """
    Effectue une recherche DFS sur l'image à partir des coordonnées spécifiées.
    """
    stack = [(x, y)]

    while stack:
        x, y = stack.pop()

        # vérifier si le pixel actuel a déjà été visité ou s'il n'est pas blanc
        if (x, y) in visited or not (
            0 <= x < image.shape[1] and 0 <= y < image.shape[0] and image[y, x] == 255
        ):
            continue

        # marquer le pixel actuel comme visité et l'ajouter au contour actuel
        visited.add((x, y))
        contours[-1].append((x, y))

        # ajouter les voisins du pixel actuel à la pile
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                stack.append((x + dx, y + dy))


def find_contours(image, min_contour_area=1000):
    """
    Trouve tous les pixels connectés dans l'image et les retourne sous forme de liste de contours.
    """
    contours = []
    visited = set()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # vérifier que le pixel actuel n'a pas été visité et qu'il est blanc
            if (x, y) not in visited and image[y, x] == 255:
                contours.append([])
                dfs(x, y, image, visited, contours)

    return [c for c in contours if len(c) > min_contour_area]


def calculate_center(contours):
    if not contours:
        return None  # Aucun contour trouvé

    upper_x, upper_y, lower_x, lower_y = contours

    center_x = (upper_x + lower_x) // 2
    center_y = (upper_y + lower_y) // 2

    return center_x, center_y


def in_range_detect(image, lower_bound, upper_bound):
    """
    Effectue une détection de couleur dans la plage spécifiée et retourne un masque
    et les coordonnées supérieures et inférieures de l'objet détecté.

    :param image: l'image à traiter
    :param lower_bound: la limite inférieure pour la détection de couleur
    :param upper_bound: la limite supérieure pour la détection de couleur

    :return: un masque avec l'objet détecté et les coordonnées supérieures et inférieures de l'objet détecté
    """
    if len(image.shape) == 2:
        raise ValueError("L'image d'entrée doit être au format HSV (3 dimensions)")

    # Obtenir la hauteur et la largeur de l'image
    height, width, _ = image.shape

    # Initialiser un masque de sortie avec des zéros
    mask = np.zeros((height, width), dtype=np.uint8)

    # Extraire les limites inférieure et supérieure pour chaque canal
    lower_bound_b, lower_bound_g, lower_bound_r = lower_bound
    upper_bound_b, upper_bound_g, upper_bound_r = upper_bound

    upper_x = 0
    upper_y = 0
    lower_x = width
    lower_y = height
    for y in range(height):
        for x in range(width):
            # Extraire les valeurs HSV du pixel actuel
            h, s, v = image[y, x]

            # Vérifier si les valeurs du pixel sont dans la plage spécifiée pour chaque canal
            if (
                lower_bound_b <= h <= upper_bound_b
                and lower_bound_g <= s <= upper_bound_g
                and lower_bound_r <= v <= upper_bound_r
            ):
                mask[y, x] = 255  # Mettre à 255 s'il est dans la plage

                # mettre à jour les coordonnées supérieures et inférieures
                if x > upper_x:
                    upper_x = x
                if y > upper_y:
                    upper_y = y
                if x < lower_x:
                    lower_x = x
                if y < lower_y:
                    lower_y = y

    # obtenir les coordonnées supérieures et inférieures
    if upper_x == 0 and upper_y == 0 and lower_x == width and lower_y == height:
        return mask, None

    return mask, (upper_x, upper_y, lower_x, lower_y)

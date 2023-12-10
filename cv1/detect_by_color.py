import cv2
import numpy as np
import cv1.detection as detection


def detect_objects_by_color(image, target_color_lower, target_color_upper):
    """
    Détecte les objets dans une image en utilisant la détection de couleur.

    Args:
    - image: Chemin de l'image ou image numpy.
    - target_color_lower: Limite inférieure de la couleur cible pour la détection.
    - target_color_upper: Limite supérieure de la couleur cible pour la détection.

    Returns:
    - None
    """
    # Charger l'image depuis le chemin spécifié
    image = cv2.imread(image)

    # Convertir l'image en espace de couleur HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(target_color_lower)
    upper_bound = np.array(target_color_upper)

    # Effectuer la détection de couleur avec les bornes spécifiées
    _, contour = detection.in_range_detect(hsv_image, lower_bound, upper_bound)

    # Créer une copie de l'image originale pour afficher les résultats
    final = image.copy()
    if contour:
        # Dessiner un rectangle autour de l'objet détecté
        cv2.rectangle(
            final, (contour[0], contour[1]), (contour[2], contour[3]), (0, 255, 0), 2
        )

    # Afficher l'image originale et l'image avec les objets détectés
    cv2.imshow("Image originale", image)
    cv2.imshow("Objets détectés", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_objects_by_color_upgraded(image, target_color_lower, target_color_upper):
    """
    Détecte les objets dans une image en utilisant la détection de couleur améliorée.

    Args:
    - image: Chemin de l'image ou image numpy.
    - target_color_lower: Limite inférieure de la couleur cible pour la détection.
    - target_color_upper: Limite supérieure de la couleur cible pour la détection.

    Returns:
    - Image avec les objets détectés et entourés par des rectangles.
    """
    # Charger l'image depuis le chemin spécifié
    image = cv2.imread(image)

    # Convertir l'image en espace de couleur HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(target_color_lower)
    upper_bound = np.array(target_color_upper)

    # Effectuer la détection de couleur avec les bornes spécifiées
    mask, _ = detection.in_range_detect(hsv_image, lower_bound, upper_bound)

    # Créer une copie de l'image originale pour afficher les résultats
    final = image.copy()

    # Trouver les contours des objets détectés
    contours = detection.find_contours(mask)

    # Dessiner un rectangle autour de chaque objet détecté
    for c in contours:
        x, y, w, h = cv2.boundingRect(np.array(c))
        cv2.rectangle(final, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Renvoyer l'image avec les objets détectés
    return final

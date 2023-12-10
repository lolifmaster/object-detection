import cv2
import numpy as np
from cv1 import tools


def green_screen_image(*, img, background_img, lower_green=None, upper_green=None):
    """
    Applique un effet de fond vert à une image statique.

    Args:
        img (str): Chemin vers l'image d'origine.
        background_img (str): Chemin vers l'image de fond.
        lower_green (Optional[np.ndarray]): Limite inférieure de la plage de couleur verte en format HSV.
        upper_green (Optional[np.ndarray]): Limite supérieure de la plage de couleur verte en format HSV.
    """
    if not lower_green:
        lower_green = np.array([0, 120, 70])
    if not upper_green:
        upper_green = np.array([10, 255, 255])

    # Charger l'image
    image = cv2.imread(img)
    # Convertir l'image de l'espace couleur BGR à HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Créer des masques binaires pour la plage de couleurs spécifiée
    color_mask = tools.in_range(hsv_image, lower_green, upper_green)
    # Extraire le premier plan (objet) de l'image
    foreground = tools.bitwise_and(image, mask=color_mask)

    # Mettre une image de fond
    background = cv2.imread(background_img)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    background = tools.bitwise_and(background, mask=tools.bitwise_not(color_mask))
    foreground = tools.add_weighted(foreground, 1, background, 1, 0)

    # Afficher le résultat
    cv2.imshow("Fond Vert", foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def green_screen_realtime(*, lower_green=None, upper_green=None, background_img=None):
    """
    Applique un effet de fond vert en temps réel à partir de la webcam.

    Args:
        lower_green (Optional[np.ndarray]): Limite inférieure de la plage de couleur verte en format HSV.
        upper_green (Optional[np.ndarray]): Limite supérieure de la plage de couleur verte en format HSV.
        background_img (Optional[str]): Chemin vers l'image de fond. Si non spécifié, utilise la webcam.
    """
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not lower_green:
        lower_green = np.array([0, 120, 70])
    if not upper_green:
        upper_green = np.array([10, 255, 255])
    if background_img is None:
        _, background = cap.read()
        cv2.flip(background, 1, background)
    else:
        background = cv2.imread(background_img)
        background = cv2.resize(background, (320, 240))

    while True:
        # Capturer l'image actuelle
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        # Convertir l'image de l'espace couleur BGR à HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Créer des masques binaires pour la plage de couleurs spécifiée
        color_mask = tools.in_range(hsv_frame, lower_green, upper_green)
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=10
        )
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1
        )
        # Extraire le premier plan (objet) de l'image
        foreground = tools.bitwise_and(frame, mask=color_mask)

        # Mettre une image de fond
        current_background = tools.bitwise_and(
            background, mask=tools.bitwise_not(color_mask)
        )
        foreground = tools.add_weighted(foreground, 1, current_background, 1, 0)

        # Afficher le résultat en temps réel
        cv2.imshow("Fond Vert ", foreground)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libérer la webcam et fermer toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()

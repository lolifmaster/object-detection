import cv2
import numpy as np
from cv1 import tools, detection


def green_screen_image(*, img, background_img, lower_green=None, upper_green=None):
    """
    Remplace l'arrière-plan vert d'une image par un autre arrière-plan.

    Args:
        img (str): Chemin vers l'image d'entrée avec un arrière-plan vert.
        background_img (str): Chemin vers l'image d'arrière-plan à utiliser.
        lower_green (np.array): Seuil HSV inférieur pour la couleur verte (par défaut [0, 120, 70]).
        upper_green (np.array): Seuil HSV supérieur pour la couleur verte (par défaut [10, 255, 255]).
    """
    # Charger l'image d'entrée et la convertir de l'espace couleur BGR à HSV
    image = cv2.imread(img)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Créer un masque binaire pour la plage de couleur verte
    color_mask = tools.in_range(hsv_image, lower_green, upper_green)

    # Extraire le premier plan (objet) de l'image
    foreground = tools.bitwise_and(image, mask=color_mask)

    # Charger l'image d'arrière-plan, la redimensionner et créer un masque pour les zones non vertes
    background = cv2.imread(background_img)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    background = tools.bitwise_and(background, mask=tools.bitwise_not(color_mask))

    # Combiner le premier plan et l'arrière-plan avec des poids
    foreground = tools.add_weighted(foreground, 1, background, 1, 0)

    # Afficher le résultat
    cv2.imshow("Chromakey", foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def green_screen_realtime(
    *, lower_green=None, upper_green=None, background_img=None, mode="pixel"
):
    """
    Réalise des effets de fond vert en temps réel à partir d'une webcam.

    Args:
        lower_green (np.array): Seuil HSV inférieur pour la couleur verte (par défaut [0, 120, 70]).
        upper_green (np.array): Seuil HSV supérieur pour la couleur verte (par défaut [10, 255, 255]).
        background_img (str): Chemin vers l'image d'arrière-plan pour le remplacement en temps réel.
        mode (str): Mode de remplacement de l'arrière-plan ["pixel", "contour"] (par défaut "pixel").
    """
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Utiliser les valeurs par défaut si les seuils verts ne sont pas spécifiés
    if not lower_green:
        lower_green = np.array([0, 120, 70])
    if not upper_green:
        upper_green = np.array([10, 255, 255])

    # Capturer une image de fond si aucune image d'arrière-plan n'est spécifiée
    if background_img is None:
        _, background = cap.read()
    else:
        background = cv2.imread(background_img)
        background = cv2.resize(background, (320, 240))

    while True:
        # Capturer le frame actuel depuis la webcam
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)

        # Convertir le frame de l'espace couleur BGR à HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if mode == "pixel":
            # Créer un masque binaire pour la plage de couleur verte
            color_mask = tools.in_range(hsv_frame, lower_green, upper_green)

            # Extraire le premier plan (objet) du frame
            foreground = tools.bitwise_and(frame, mask=color_mask)

            # Créer un masque pour les zones non vertes
            background_mask = tools.bitwise_not(color_mask)

            # Combiner le premier plan et l'arrière-plan avec des poids
            masked_background = tools.bitwise_and(background, mask=background_mask)
            final = tools.add_weighted(foreground, 1, masked_background, 1, 0)

        elif mode == "contour":
            # Créer un masque binaire pour la plage de couleur verte
            _, contour = detection.in_range_detect(hsv_frame, lower_green, upper_green)

            # Copier l'image d'arrière-plan et remplacer la région spécifiée par le frame actuel
            final = background.copy()
            if contour:
                upper_x, upper_y, lower_x, lower_y = contour
                final[lower_y:upper_y, lower_x:upper_x] = frame[
                    lower_y:upper_y, lower_x:upper_x
                ]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Afficher le résultat en temps réel
        cv2.imshow("Green Screen", final)

        # update the background image if the user presses 'b'
        if cv2.waitKey(1) & 0xFF == ord("b"):
            _, background = cap.read()
            cv2.flip(background, 1, background)

        # Quitter la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libérer la webcam et fermer toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()

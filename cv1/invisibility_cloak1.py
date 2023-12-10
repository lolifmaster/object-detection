import cv2
import numpy as np
from cv1 import tools, detection


def invisibility_cloak(*, lower_red, upper_red, background_img=None):
    """
    Applique l'effet de manteau d'invisibilité en utilisant la détection de couleur.

    Args:
        lower_red (List[int]): Seuil inférieur de la plage de couleur rouge en format HSV.
        upper_red (List[int]): Seuil supérieur de la plage de couleur rouge en format HSV.
        background_img (Optional[str]): Chemin vers l'image de fond. Si non spécifié, utilise la webcam.
    """
    if not lower_red:
        lower_red = np.array([0, 120, 70])
    if not upper_red:
        upper_red = np.array([10, 255, 255])

    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Capturer l'image de fond
    if background_img is not None:
        background_img = cv2.imread(background_img)
        background = cv2.resize(background_img, (320, 240))
    else:
        _, background = cap.read()
        cv2.flip(background, 1, background)

    while True:
        # Capturer l'image actuelle
        ret, frame = cap.read()
        cv2.flip(frame, 1)
        # Convertir l'image de l'espace couleur BGR à HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masque_couleur, contour = detection.in_range_detect(
            hsv_frame, lower_red, upper_red
        )

        if contour:
            upper_x, upper_y, lower_x, lower_y = contour
            frame[lower_y:upper_y, lower_x:upper_x] = background[
                lower_y:upper_y, lower_x:upper_x
            ]

        cv2.imshow("Manteau d'Invisibilité", frame)

        # Interrompre la boucle si la touche 'q' est pressée
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Libérer la webcam et fermer toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()

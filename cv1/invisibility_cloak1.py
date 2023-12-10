import cv2
import numpy as np
from cv1 import tools, detection


def invisibility_cloak(*, seuil_rouge_inf, seuil_rouge_sup, image_fond=None):
    """
    Applique l'effet de manteau d'invisibilité en utilisant la détection de couleur.

    Args:
        seuil_rouge_inf (List[int]): Seuil inférieur de la plage de couleur rouge en format HSV.
        seuil_rouge_sup (List[int]): Seuil supérieur de la plage de couleur rouge en format HSV.
        image_fond (Optional[str]): Chemin vers l'image de fond. Si non spécifié, utilise la webcam.
    """
    if not seuil_rouge_inf:
        seuil_rouge_inf = np.array([0, 120, 70])
    if not seuil_rouge_sup:
        seuil_rouge_sup = np.array([10, 255, 255])

    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Capturer l'image de fond
    if image_fond is not None:
        image_fond = cv2.imread(image_fond)
        background = cv2.resize(image_fond, (320, 240))
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
            hsv_frame, seuil_rouge_inf, seuil_rouge_sup
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

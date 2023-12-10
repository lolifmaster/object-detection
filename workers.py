from PyQt5.QtCore import QThread, pyqtSignal
from game import CarDodgingGame
from cv1 import (
    invisibility_cloak,
    green_screen_realtime,
    detect_objects_by_color_real_time,
)


class GameHandler(QThread):
    """
    Fil pour exécuter le jeu Car Dodging.
    Émet un signal lorsque le jeu est terminé.
    """

    game_finished = pyqtSignal()

    def __init__(self, parent=None, camera=True):
        """
        Initialise le thread GameHandler.

        Paramètres :
        - parent : QObject parent.
        - camera : Drapeau indiquant s'il faut utiliser une caméra pour le jeu.
        """
        super().__init__(parent)
        self.game = CarDodgingGame(camera=camera)

    def run(self):
        """
        Exécute le jeu Car Dodging dans un thread séparé.
        Émet le signal game_finished lorsque le jeu est terminé.
        """
        try:
            self.game.run()
            self.game_finished.emit()
        except Exception as e:
            print(f"Erreur dans GameHandler : {e}")


class InvisibilityCloakThread(QThread):
    """
    Fil pour implémenter l'effet Cape d'Invisibilité.
    """

    def __init__(self, lower_red, upper_red, background_img, parent=None):
        """
        Initialise le thread InvisibilityCloakThread.

        Paramètres :
        - lower_red : Limite inférieure de la couleur rouge pour la cape d'invisibilité.
        - upper_red : Limite supérieure de la couleur rouge pour la cape d'invisibilité.
        - background_img : Image d'arrière-plan pour l'effet de la cape d'invisibilité.
        - parent : QObject parent.
        """
        super().__init__(parent)
        self.lower_red = lower_red
        self.upper_red = upper_red
        self.background_img = background_img

    def run(self):
        """
        Exécute l'effet de la cape d'invisibilité dans un thread séparé.
        """
        try:
            invisibility_cloak(
                lower_red=self.lower_red,
                upper_red=self.upper_red,
                background_img=self.background_img,
            )
        except Exception as e:
            print(f"Erreur dans InvisibilityCloakThread : {e}")


class GreenScreenThread(QThread):
    """
    Fil pour implémenter l'effet de l'écran vert.
    """

    def __init__(
        self, lower_green, upper_green, background_img, parent=None, mode="pixel"
    ):
        """
        Initialise le thread GreenScreenThread.

        Paramètres :
        - lower_green : Limite inférieure de la couleur verte pour l'effet de l'écran vert.
        - upper_green : Limite supérieure de la couleur verte pour l'effet de l'écran vert.
        - background_img : Image d'arrière-plan pour l'effet de l'écran vert.
        - parent : QObject parent.
        """
        super().__init__(parent)
        self.lower_green = lower_green
        self.upper_green = upper_green
        self.background_img = background_img
        self.mode = mode

    def run(self):
        """
        Exécute l'effet de l'écran vert dans un thread séparé.
        """
        try:
            green_screen_realtime(
                lower_green=self.lower_green,
                upper_green=self.upper_green,
                background_img=self.background_img,
                mode=self.mode,
            )
        except Exception as e:
            print(f"Erreur dans GreenScreenThread : {e}")


class ObjectDetectionThread(QThread):
    """
    Fil pour effectuer la détection d'objets en temps réel.
    """

    def __init__(self, parent=None):
        """
        Initialise le thread ObjectDetectionThread.

        Paramètres :
        - parent : QObject parent.
        """
        super().__init__(parent)

    def run(self):
        """
        Exécute la détection d'objets en temps réel dans un thread séparé.
        """
        try:
            detect_objects_by_color_real_time()
        except Exception as e:
            print(f"Erreur dans ObjectDetectionThread : {e}")

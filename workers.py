from PyQt5.QtCore import QThread, pyqtSignal
from game import CarDodgingGame
from cv1 import (
    invisibility_cloak,
    green_screen_realtime,
    detect_objects_by_color_real_time,
)


class GameHandler(QThread):
    game_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.game = CarDodgingGame(camera=True)

    def run(self):
        try:
            self.game.start()
            self.game_finished.emit()
        except Exception as e:
            print(f"Error in GameHandler: {e}")


class InvisibilityCloakThread(QThread):
    def __init__(self, lower_red, upper_red, background_img, parent=None):
        super().__init__(parent)
        self.lower_red = lower_red
        self.upper_red = upper_red
        self.background_img = background_img

    def run(self):
        try:
            invisibility_cloak(
                lower_red=self.lower_red,
                upper_red=self.upper_red,
                background_img=self.background_img,
            )
        except Exception as e:
            print(f"Error in InvisibilityCloakThread: {e}")


class GreenScreenThread(QThread):
    def __init__(self, lower_green, upper_green, background_img, parent=None):
        super().__init__(parent)
        self.lower_green = lower_green
        self.upper_green = upper_green
        self.background_img = background_img

    def run(self):
        try:
            green_screen_realtime(
                lower_green=self.lower_green,
                upper_green=self.upper_green,
                background_img=self.background_img,
            )
        except Exception as e:
            print(f"Error in GreenScreenThread: {e}")


class ObjectDetectionThread(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            detect_objects_by_color_real_time()
        except Exception as e:
            print(f"Error in ObjectDetectionThread: {e}")

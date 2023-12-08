from PyQt5.QtCore import QThread, pyqtSignal
from game import CarDodgingGame


class GameHandler(QThread):
    game_finished = pyqtSignal()

    def __init__(self, parent=None):
        super(GameHandler, self).__init__(parent)
        self.game = CarDodgingGame()

    def run(self):
        self.game.run()
        self.game_finished.emit()

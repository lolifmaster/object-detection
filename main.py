from gui import ImageFilterApp
import sys
from PyQt5.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageFilterApp()
    sys.exit(app.exec_())

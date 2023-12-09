import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
    QDialog,
)
from PyQt5.QtGui import QPixmap, QImage
import cv2
from cv1 import tools
from game_worker import GameHandler
from image_processor import FilterInputDialog, FILTERS


class ImageFilterApp(QWidget):
    def __init__(self):
        super().__init__()

        self.game_button = None
        self.apply_button = None
        self.upload_button = None
        self.image_label = None
        self.filter_combobox = None
        self.original_image = None
        self.game_thread = None
        self.filters = FILTERS
        self.init_ui()

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.filter_combobox = QComboBox(self)
        for filter_data in self.filters:
            self.filter_combobox.addItem(filter_data["name"])

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.load_image)

        self.apply_button = QPushButton("Apply Filter", self)
        self.apply_button.clicked.connect(self.show_filter_input_dialog)

        self.game_button = QPushButton("Play Game", self)
        self.game_button.clicked.connect(self.start_game_thread)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.upload_button)

        # Improved layout using QHBoxLayout
        hbox = QHBoxLayout()
        hbox.addWidget(self.filter_combobox)
        hbox.addWidget(self.apply_button)
        hbox.addWidget(self.game_button)
        vbox.addLayout(hbox)

        # Add stretch factor to the image_label
        vbox.addWidget(self.image_label, 1)

        # Add spacer to push widgets to the top
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vbox.addItem(spacer)

        self.setWindowTitle("Image Filter App")
        self.setGeometry(100, 100, 800, 600)

        self.setLayout(vbox)
        self.show()

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jfif)"
        )
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

            # Save the original image for later use
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def show_filter_input_dialog(self):
        # Check if an image is loaded
        if self.original_image is None:
            self.show_error_message("Please upload an image before applying a filter.")
            return

        selected_filter_index = self.filter_combobox.currentIndex()
        selected_filter = self.filters[selected_filter_index]
        test_image = self.original_image

        # apply threshold to image if selected filter is opening or closing or erosion or dilation
        if selected_filter["name"] in ["Opening", "Closing", "Erosion", "Dilation"]:
            test_image = tools.threshold(self.original_image, 127, 255)

        # Show the input dialog only if the selected filter has arguments
        if selected_filter["arguments"]:
            input_dialog = FilterInputDialog(selected_filter["arguments"], self)
            result = input_dialog.exec_()

            if result == QDialog.Accepted:
                argument_values = input_dialog.get_argument_values()

                # Apply the selected filter to the original image with the provided arguments
                result_image = selected_filter["function"](
                    test_image, **argument_values
                )
            else:
                return

        else:
            # Apply the selected filter to the original image without arguments
            result_image = selected_filter["function"](test_image)
        # Display the result image
        self.display_image(result_image)

    def show_error_message(self, message):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.exec_()

    def display_image(self, image_data):
        # Convert the image data to a format that can be displayed with QPixmap
        height, width = image_data.shape
        bytes_per_line = width
        image = QImage(
            image_data, width, height, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap(image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def start_game_thread(self):
        if self.game_thread is not None and self.game_thread.isRunning():
            self.show_error_message("The game is already running!")
            return
        self.game_button.setEnabled(False)
        self.game_thread = GameHandler()
        self.game_thread.game_finished.connect(self.on_game_finished)
        self.game_thread.start()

    def on_game_finished(self):
        QMessageBox.information(self, "Game Finished", "The game has finished!")
        self.game_thread.quit()
        self.game_thread = None
        self.game_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageFilterApp()
    sys.exit(app.exec_())

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
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
from cv1 import tools
from workers import (
    GameHandler,
    InvisibilityCloakThread,
    GreenScreenThread,
    ObjectDetectionThread,
)
from image_processor import FilterInputDialog, FILTERS
from cv1 import (
    green_screen_realtime,
    detect_objects_by_color_real_time,
    invisibility_cloak,
    detect_objects_by_color_upgraded,
)
from multiprocessing import Process


class ImageFilterApp(QWidget):
    def __init__(self):
        super().__init__()

        self.detect_object_button = None
        self.invisibility_cloak = None
        self.green_screen_button = None
        self.detect_button = None
        self.game_button = None
        self.apply_button = None
        self.upload_button = None
        self.image_label = None
        self.filter_combobox = None
        self.original_image = None
        self.game_thread = None
        self.invisibility_cloak_thread = None
        self.green_screen_thread = None
        self.object_detection_thread = None
        self.original_image_path = None
        self.filters = FILTERS
        self.init_ui()

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Align image in the center
        self.image_label.setAlignment(Qt.AlignCenter)

        self.filter_combobox = QComboBox(self)
        for filter_data in self.filters:
            self.filter_combobox.addItem(filter_data["name"])

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.load_image)

        self.apply_button = QPushButton("Apply Filter", self)
        self.apply_button.clicked.connect(self.show_filter_input_dialog)

        self.game_button = QPushButton("Play Game", self)
        self.game_button.clicked.connect(self.start_game_thread)

        self.detect_button = QPushButton("Detect Video", self)
        self.detect_button.clicked.connect(self.apply_object_detection)

        self.detect_object_button = QPushButton("Detect Objects", self)
        self.detect_object_button.clicked.connect(self.detect_objects_by_color)

        self.green_screen_button = QPushButton("Green Screen", self)
        self.green_screen_button.clicked.connect(
            self.apply_green_screen_real_time)

        self.invisibility_cloak = QPushButton("Invisibility Cloak", self)
        self.invisibility_cloak.clicked.connect(self.apply_invisibility_cloak)

        # Adding labels
        upload_label = QLabel("Upload Image:", self)
        filter_label = QLabel("Select Filter:", self)
        realtime_label = QLabel("Detection Features:", self)

        vbox = QVBoxLayout(self)
        vbox.addWidget(upload_label)
        vbox.addWidget(self.upload_button)

        vbox.addWidget(filter_label)
        vbox.addWidget(self.filter_combobox)
        vbox.addWidget(self.apply_button)

        vbox.addWidget(realtime_label)
        rt_features = QHBoxLayout()
        rt_features.addWidget(self.detect_button)
        rt_features.addWidget(self.green_screen_button)
        rt_features.addWidget(self.invisibility_cloak)
        rt_features.addWidget(self.game_button)
        rt_features.addWidget(self.detect_object_button)
        vbox.addLayout(rt_features)

        # Add stretch factor to the image_label
        vbox.addWidget(self.image_label, 1)

        # Add spacer to push widgets to the top
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum,
                             QSizePolicy.Expanding)
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
            self.original_image_path = file_path
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def show_filter_input_dialog(self):
        # Check if an image is loaded
        if self.original_image is None:
            self.show_error_message(
                "Please upload an image before applying a filter.")
            return

        selected_filter_index = self.filter_combobox.currentIndex()
        selected_filter = self.filters[selected_filter_index]
        test_image = self.original_image

        # apply threshold to image if selected filter is opening or closing or erosion or dilation
        if selected_filter["name"] in [
            "Opening Filter",
            "Closing Filter",
            "Erosion Filter",
            "Dilation Filter",
        ]:
            test_image = tools.threshold(self.original_image, 127, 255)

        # Show the input dialog only if the selected filter has arguments
        if selected_filter["arguments"]:
            input_dialog = FilterInputDialog(
                selected_filter["arguments"], self)
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

    def display_image_color(self, image_data):
        # Convert the image data to a format that can be displayed with QPixmap
        height, width, _ = image_data.shape
        bytes_per_line = width * 3
        image = QImage(image_data, width, height,
                       bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap(image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def apply_invisibility_cloak(self):
        self.invisibility_cloak_thread = InvisibilityCloakThread(
            lower_red=[0, 120, 70],
            upper_red=[10, 255, 255],
            background_img=self.original_image_path,
        )
        self.invisibility_cloak_thread.start()

    def apply_green_screen_real_time(self):
        self.green_screen_thread = GreenScreenThread(
            lower_green=[0, 120, 70],
            upper_green=[10, 255, 255],
            background_img=self.original_image_path,
        )
        self.green_screen_thread.start()

    def apply_object_detection(self):
        self.object_detection_thread = ObjectDetectionThread()
        self.object_detection_thread.start()

    def detect_objects_by_color(self):
        if self.original_image_path is None:
            self.show_error_message(
                "Please upload an image before detecting...")
            return
        detected_img = detect_objects_by_color_upgraded(
            image=self.original_image_path,
            target_color_lower=[0, 120, 70],
            target_color_upper=[10, 255, 255],
        )
        self.display_image_color(detected_img)

    def start_game_thread(self):
        if self.game_thread is not None and self.game_thread.isRunning():
            self.show_error_message("The game is already running!")
            return
        self.game_button.setEnabled(False)
        self.game_thread = GameHandler()
        self.game_thread.game_finished.connect(self.on_game_finished)
        self.game_thread.start()

    def on_game_finished(self):
        QMessageBox.information(self, "Game Finished",
                                "The game has finished!")
        self.game_thread.quit()
        self.game_thread = None
        self.game_button.setEnabled(True)

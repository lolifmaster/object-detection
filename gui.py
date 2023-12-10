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
    """
    The main window of the application.
    """

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
        self.detection_feature_running = False
        self.init_ui()

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap("data/placeholder.jpg"))

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
        self.green_screen_button.clicked.connect(self.apply_green_screen_real_time)

        self.invisibility_cloak = QPushButton("Invisibility Cloak", self)
        self.invisibility_cloak.clicked.connect(self.apply_invisibility_cloak)

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

        vbox.addWidget(self.image_label, 1)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vbox.addItem(spacer)

        self.setWindowTitle("Image Filter App")
        self.setGeometry(0, 0, 1000, 800)

        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        self.move(x, 20)

        self.setLayout(vbox)
        self.show()

    def load_image(self):
        """
        Charge une image à partir du fichier sélectionné par l'utilisateur
        et l'affiche dans l'interface graphique.
        """
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
        """
        Affiche une boîte de dialogue permettant à l'utilisateur de
        configurer les paramètres du filtre sélectionné avant de l'appliquer
        à l'image d'origine.
        """
        if self.original_image is None:
            self.show_error_message("Please upload an image before applying a filter.")
            return

        selected_filter_index = self.filter_combobox.currentIndex()
        selected_filter = self.filters[selected_filter_index]
        test_image = self.original_image

        if selected_filter["name"] in [
            "Opening Filter",
            "Closing Filter",
            "Erosion Filter",
            "Dilation Filter",
        ]:
            test_image = tools.threshold(self.original_image, 127, 255)

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
            result_image = selected_filter["function"](test_image)
        self.display_image(result_image)

    def show_error_message(self, message):
        """
        Affiche une boîte de dialogue d'erreur avec le message spécifié.
        """
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.exec_()

    def display_image(self, image_data):
        """
        Affiche une image dans l'interface graphique à partir des données
        de l'image spécifiées.
        """
        height, width = image_data.shape
        bytes_per_line = width
        image = QImage(
            image_data, width, height, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap(image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def display_image_color(self, image_data):
        """
        Affiche une image couleur dans l'interface graphique à partir des
        données de l'image spécifiées.
        """

        height, width, _ = image_data.shape
        bytes_per_line = width * 3
        image = QImage(image_data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap(image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def apply_invisibility_cloak(self):
        """
        Applique l'effet de la cape d'invisibilité à l'image en temps réel
        en utilisant un thread dédié.
        """
        if self.detection_feature_running:
            self.show_error_message(
                "Cannot start Invisibility Cloak while another feature is running."
            )
            return

        self.detection_feature_running = True
        self.invisibility_cloak_thread = InvisibilityCloakThread(
            lower_red=[0, 120, 70],
            upper_red=[10, 255, 255],
            background_img=self.original_image_path,
        )
        self.invisibility_cloak_thread.finished.connect(
            self.on_detection_feature_finished
        )
        self.invisibility_cloak_thread.start()

    def apply_green_screen_real_time(self):
        """
        Applique l'effet d'écran vert à l'image en temps réel en utilisant
        un thread dédié.
        """
        if self.detection_feature_running:
            self.show_error_message(
                "Cannot start Green Screen while another feature is running."
            )
            return

        # select the green screen mode
        dialog = QDialog(self)
        dialog.setWindowTitle("Green Screen Options")

        label = QLabel("Which mode do you want to use?")
        pixels_button = QPushButton("Pixels")
        no_button = QPushButton("Contour")

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(pixels_button)
        layout.addWidget(no_button)
        dialog.setLayout(layout)

        pixels_button.clicked.connect(lambda: self.start_green_screen("pixel", dialog))
        no_button.clicked.connect(lambda: self.start_green_screen("contour", dialog))

        # Show the dialog
        dialog.exec_()

    def start_green_screen(self, mode, dialog):
        """
        Démarre le thread d'écran vert en fonction de la décision de
        l'utilisateur d'utiliser les pixels ou les contours.
        """
        dialog.accept()
        self.detection_feature_running = True
        self.green_screen_thread = GreenScreenThread(
            lower_green=[0, 120, 70],
            upper_green=[10, 255, 255],
            background_img=self.original_image_path,
            mode=mode,
        )
        self.green_screen_thread.finished.connect(self.on_detection_feature_finished)
        self.green_screen_thread.start()

    def apply_object_detection(self):
        """
        Lance un thread dédié pour effectuer la détection d'objets dans
        l'image en temps réel.
        """
        if self.detection_feature_running:
            self.show_error_message(
                "Cannot start Object Detection while another feature is running."
            )
            return

        self.detection_feature_running = True
        self.object_detection_thread = ObjectDetectionThread()
        self.object_detection_thread.finished.connect(
            self.on_detection_feature_finished
        )
        self.object_detection_thread.start()

    def on_detection_feature_finished(self):
        """
        Méthode appelée lorsque la fonctionnalité de détection actuelle est
        terminée. Réinitialise l'état de détection.
        """
        self.detection_feature_running = False
        QMessageBox.information(
            self, "Detection Feature Finished", "The detection feature has finished!"
        )

    def detect_objects_by_color(self):
        """
        Effectue la détection d'objets dans l'image en fonction des couleurs
        spécifiées.
        """
        if self.original_image_path is None:
            self.show_error_message("Please upload an image before detecting...")
            return
        detected_img = detect_objects_by_color_upgraded(
            image=self.original_image_path,
            target_color_lower=[0, 120, 70],
            target_color_upper=[10, 255, 255],
        )
        self.display_image_color(detected_img)

    def start_game_thread(self):
        """
        Lance un dialogue pour permettre à l'utilisateur de choisir s'il
        souhaite utiliser la caméra pour jouer à un jeu.
        """
        if self.game_thread is not None and self.game_thread.isRunning():
            self.show_error_message("Wait for the current thread to end!")
            return

        if self.detection_feature_running:
            self.show_error_message(
                "Cannot start the game while another feature is running."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Game Options")

        label = QLabel("Do you want to use the camera for the game?")
        yes_button = QPushButton("Yes")
        no_button = QPushButton("No")

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(yes_button)
        layout.addWidget(no_button)
        dialog.setLayout(layout)

        yes_button.clicked.connect(lambda: self.start_game(True, dialog))
        no_button.clicked.connect(lambda: self.start_game(False, dialog))

        # Show the dialog
        dialog.exec_()

    def start_game(self, use_camera, dialog):
        """
        Démarre le thread du jeu en fonction de la décision de l'utilisateur
        de l'utiliser avec ou sans la caméra.
        """
        dialog.accept()  # Close the dialog
        self.detection_feature_running = True

        if self.game_thread is not None and self.game_thread.isRunning():
            self.show_error_message("The game is already running!")
            return

        self.game_button.setEnabled(False)
        self.game_thread = GameHandler(camera=use_camera)
        self.game_thread.game_finished.connect(self.on_game_finished)
        self.game_thread.start()

    def on_game_finished(self):
        """
        Méthode appelée lorsque le jeu est terminé. Affiche une boîte de
        dialogue informant que le jeu est terminé.
        """
        QMessageBox.information(self, "Game Finished", "The game has finished!")
        self.game_thread.quit()
        self.game_thread = None
        self.game_button.setEnabled(True)
        self.detection_feature_running = False

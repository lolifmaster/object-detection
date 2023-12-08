import sys
import ast
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QLineEdit,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage
import cv2
from cv1.filters import (
    mean,
    median,
    gaussian,
    laplacian,
    edge_detection,
    sharpen,
    emboss,
    bilateral,
    opening,
    closing,
    erode,
    dilate,
)
from cv1 import Shape, tools


class FilterInputDialog(QDialog):
    def __init__(self, arguments, parent=None):
        super(FilterInputDialog, self).__init__(parent)
        self.setWindowTitle("Filter Parameters")
        self.layout = QFormLayout(self)
        self.arguments = arguments
        self.input_widgets = {}
        self.error_labels = {}

        for argument_name in self.arguments:
            if argument_name == "kernel_shape":
                input_widget = QComboBox(self)
                input_widget.addItems(["rect", "cross"])
            else:
                input_widget = QLineEdit(self)
            error_label = QLabel(self)
            error_label.setStyleSheet("color: red; font-size: 10px;")
            self.layout.addRow(f"{argument_name.capitalize()}: ", input_widget)
            self.layout.addRow("", error_label)
            self.input_widgets[argument_name] = input_widget
            self.error_labels[argument_name] = error_label

        self.submit_button = QPushButton("Apply Filter", self)
        self.submit_button.clicked.connect(self.validate_and_accept)
        self.layout.addRow(self.submit_button)

    def validate_and_accept(self):
        if self.validate_inputs():
            self.accept()

    def validate_inputs(self):
        for argument_name, input_widget in self.input_widgets.items():
            error_label = self.error_labels[argument_name]
            if argument_name == "kernel_shape":
                # No need to check for emptiness in a combo box
                error_label.clear()
            elif input_widget.text().strip() == "":
                error_label.setText("This field is required")
                return False
            else:
                error_label.clear()
        return True

    def get_argument_values(self):
        argument_values = {}
        for argument_name, input_widget in self.input_widgets.items():
            if argument_name == "kernel_shape":
                argument_values[argument_name] = Shape(
                    self.input_widgets[argument_name].currentText()
                )
            else:
                try:
                    argument_values[argument_name] = ast.literal_eval(
                        input_widget.text()
                    )
                except (SyntaxError, ValueError):
                    argument_values[argument_name] = input_widget.text()
        return argument_values


class ImageFilterApp(QWidget):
    def __init__(self):
        super().__init__()

        self.apply_button = None
        self.upload_button = None
        self.image_label = None
        self.filter_combobox = None
        self.original_image = None
        self.filters = [
            {"name": "Mean Filter", "function": mean, "arguments": ["kernel_size"]},
            {"name": "Median Filter", "function": median, "arguments": ["kernel_size"]},
            {
                "name": "Gaussian Filter",
                "function": gaussian,
                "arguments": ["kernel_size", "sigma"],
            },
            {"name": "Laplacian Filter", "function": laplacian, "arguments": []},
            {
                "name": "Edge Detection Filter",
                "function": edge_detection,
                "arguments": [],
            },
            {"name": "Sharpen Filter", "function": sharpen, "arguments": []},
            {"name": "Emboss Filter", "function": emboss, "arguments": []},
            {
                "name": "Bilateral Filter",
                "function": bilateral,
                "arguments": ["kernel_size", "sigma_s", "sigma_r"],
            },
            {
                "name": "Opening",
                "function": opening,
                "arguments": ["kernel_size", "iterations", "kernel_shape"],
            },
            {
                "name": "Closing",
                "function": closing,
                "arguments": ["kernel_size", "iterations", "kernel_shape"],
            },
            {
                "name": "Erosion",
                "function": erode,
                "arguments": ["kernel_size", "iterations", "kernel_shape"],
            },
            {
                "name": "Dilation",
                "function": dilate,
                "arguments": ["kernel_size", "iterations", "kernel_shape"],
            },
        ]

        self.init_ui()

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.resize(400, 400)
        self.filter_combobox = QComboBox(self)
        for filter_data in self.filters:
            self.filter_combobox.addItem(filter_data["name"])

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.load_image)

        self.apply_button = QPushButton("Apply Filter", self)
        self.apply_button.clicked.connect(self.show_filter_input_dialog)

        vbox = QVBoxLayout()
        vbox.addWidget(self.upload_button)
        vbox.addWidget(self.filter_combobox)
        vbox.addWidget(self.apply_button)
        vbox.addWidget(self.image_label)

        self.setLayout(vbox)
        self.setWindowTitle("Image Filter App")
        self.setGeometry(100, 100, 800, 600)
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

                # Display the result image
                self.display_image(result_image)
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageFilterApp()
    sys.exit(app.exec_())

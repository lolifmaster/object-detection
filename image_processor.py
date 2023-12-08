import cv2
from cv1 import Shape, tools
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QDialog,
    QFormLayout,
    QLineEdit,
)
import ast
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


FILTERS = [
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

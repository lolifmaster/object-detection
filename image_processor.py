from cv1 import Shape
from PyQt5.QtWidgets import (
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
    sobel,
    opening,
    closing,
    erode,
    dilate,
)
from PyQt5.QtCore import QTimer


class FilterInputDialog(QDialog):
    def __init__(self, arguments, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Parameters")
        self.layout = QFormLayout(self)
        self.arguments = arguments
        self.input_widgets = {}
        self.error_labels = {}
        self.setWhatsThis(DESCRIPTIONS)

        for argument_name in self.arguments:
            if argument_name == "kernel_shape":
                input_widget = QComboBox(self)
                input_widget.addItems(["rect", "cross"])
            elif argument_name == "direction":
                input_widget = QComboBox(self)
                input_widget.addItems(["x", "y", "xy"])
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

            if argument_name in ["kernel_shape", "direction"]:
                # No need to check for emptiness in a combo box
                error_label.clear()
            elif input_widget.text().strip() == "":
                error_label.setText("This field is required")
                return False
            else:
                try:
                    # Attempt to convert the input value to the expected type
                    if argument_name == "kernel_shape":
                        Shape(input_widget.currentText())
                    elif argument_name == "direction":
                        input_widget.currentText()  # No conversion needed for a combo box
                    else:
                        ast.literal_eval(input_widget.text())

                    # If the conversion succeeds, clear the error label
                    error_label.clear()
                except (SyntaxError, ValueError):
                    # If the conversion fails, set an error message
                    error_label.setText("Invalid type for this field")
                    return False

        return True

    def get_argument_values(self):
        argument_values = {}
        for argument_name, input_widget in self.input_widgets.items():
            if argument_name == "kernel_shape":
                argument_values[argument_name] = Shape(
                    self.input_widgets[argument_name].currentText()
                )
            elif argument_name == "direction":
                argument_values[argument_name] = self.input_widgets[
                    argument_name
                ].currentText()
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
        "name": "Sobel Filer",
        "function": sobel,
        "arguments": ["direction"],
        "direction": ["x", "y", "xy"],
    },
    {
        "name": "Opening Filter",
        "function": opening,
        "arguments": ["kernel_size", "iterations", "kernel_shape"],
    },
    {
        "name": "Closing Filter",
        "function": closing,
        "arguments": ["kernel_size", "iterations", "kernel_shape"],
        "kernel_shape": ["rect", "cross"],
    },
    {
        "name": "Erosion Filter",
        "function": erode,
        "arguments": ["kernel_size", "iterations", "kernel_shape"],
        "kernel_shape": ["rect", "cross"],
    },
    {
        "name": "Dilation Filter",
        "function": dilate,
        "arguments": ["kernel_size", "iterations", "kernel_shape"],
        "kernel_shape": ["rect", "cross"],
    },
]

DESCRIPTIONS = """
    "kernel_size": "The size of the kernel to use for the filter. (positive integer in morph filters).",
    "sigma": "The standard deviation of the Gaussian filter. Must be a positive number.",
    "iterations": "The number of times to apply the filter. Must be a positive integer.",
    "kernel_shape": "The shape of the kernel to use for the filter.",
    "direction": "The direction of the Sobel filter.",
"""

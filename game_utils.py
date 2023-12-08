class Taxi:
    def __init__(self) -> None:
        self.width = 50
        self.height = 80
        self.score = 1
        self.image = r"./assets/taxi_asset.png"


class Police:
    def __init__(self) -> None:
        self.width = 50
        self.height = 80
        self.score = 1
        self.image = r"assets/police_bas.png"


class Camion:
    def __init__(self) -> None:
        self.width = 70
        self.height = 160
        self.score = 2
        self.image = r"./assets/camion_asset.png"


def rgba2rgb(image):
    """
    Convert an RGBA image to RGB format.

    :param image: Input image in RGBA format.
    :return: RGB image.
    """
    if image.shape[2] == 4:
        return image[:, :, :3]
    else:
        return image

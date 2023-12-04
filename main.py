import cv1.filters as filters
from pathlib import Path
import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def main():
    # image = cv2.imread(str(DATA_DIR / "akatsuki-cat.jfif"), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(DATA_DIR / "ppp.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found")

    new_image = filters.laplacian(image)
    new_image_cv2 = cv2.Laplacian(image, cv2.CV_64F)

    cv2.imshow("original", image)
    cv2.imshow("mine", new_image)
    cv2.imshow("cv", new_image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

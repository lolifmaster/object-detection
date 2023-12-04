import cv1.tools as tools
from pathlib import Path
import cv2

DATA_DIR = Path(__file__).parent / "data"


def main():
    image = cv2.imread(str(DATA_DIR / "akatsuki-cat.jfif"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found")

    new_image = tools.filter_mean(image, 10)

    cv2.imshow("Original", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
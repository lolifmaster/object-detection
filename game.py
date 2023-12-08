import cv2
import numpy as np
import random
from cv1 import tools, detection


class CarDodgingGame:
    """
    A simple car dodging game where the player has to dodge the obstacles by moving the car left or right using the
    'a' and 'd' keys or camera input.

    :param width: the width of the game window
    :param height: the height of the game window
    :param car_width: the width of the car
    :param car_height: the height of the car
    :param obstacle_width: the width of the obstacles
    :param obstacle_height: the height of the obstacles
    :param obstacle_speed: the speed of the obstacles
    :param lower_bound: the lower bound for the color detection
    :param upper_bound: the upper bound for the color detection
    :param step: the step for moving the car

    """

    def __init__(
        self,
        width=400,
        height=600,
        car_width=50,
        car_height=30,
        obstacle_width=30,
        obstacle_height=30,
        obstacle_speed=8,
        lower_bound=np.array([90, 20, 90]),
        upper_bound=np.array([101, 38, 95]),
        step: int = 10,
    ):
        self.game_window = None
        self.width = width
        self.height = height
        self.car_width = car_width
        self.car_height = car_height
        self.obstacle_width = obstacle_width
        self.obstacle_height = obstacle_height
        self.obstacle_speed = obstacle_speed
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.car_x = width // 2 - car_width // 2
        self.STEP = step
        self.car_y = height - car_height - self.STEP
        self.obstacles = []

    def draw_car(self):
        cv2.rectangle(
            self.game_window,
            pt1=(self.car_x, self.car_y),
            pt2=(self.car_x + self.car_width, self.car_y + self.car_height),
            color=(0, 255, 0),
        )

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            cv2.rectangle(
                self.game_window,
                pt1=(obstacle[0], obstacle[1]),
                pt2=(
                    obstacle[0] + self.obstacle_width,
                    obstacle[1] + self.obstacle_height,
                ),
                color=(0, 0, 255),
            )

    def move_obstacles(self):
        self.obstacles = [
            (x, y + self.obstacle_speed)
            for x, y in self.obstacles
            if y + self.obstacle_speed < self.height
        ]

    def check_collision(self):
        for obstacle in self.obstacles:
            if (
                self.car_x < obstacle[0] + self.obstacle_width
                and self.car_x + self.car_width > obstacle[0]
                and self.car_y < obstacle[1] + self.obstacle_height
                and self.car_y + self.car_height > obstacle[1]
            ):
                return True
        return False

    def generate_obstacle(self):
        obstacle_x = random.randint(0, self.width - self.obstacle_width)
        obstacle_y = -self.obstacle_height
        self.obstacles.append((obstacle_x, obstacle_y))

    def handle_key_input(self, key):
        # Adjust the car's position based on the key input
        if key & 0xFF == ord("a") and self.car_x > 0:
            self.car_x -= self.STEP
        elif key & 0xFF == ord("d") and self.car_x < self.width - self.car_width:
            self.car_x += self.STEP

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 60)

        desired_width = 300
        desired_height = 150

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)

            self.game_window = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            hsv_frame = tools.bgr2hsv(frame)

            color_mask, contours = detection.in_range_detect(
                hsv_frame, self.lower_bound, self.upper_bound
            )

            original = tools.bitwise_and(frame, mask=color_mask)

            if contours:
                original = detection.draw_contours(
                    original, contours, color=(255, 0, 0)
                )

                center_x, _ = detection.calculate_center(contours)

                if center_x is not None:
                    self.car_x = center_x

            if random.randint(0, 100) < 10:
                self.generate_obstacle()

            self.move_obstacles()

            if self.check_collision():
                print("Game Over!")
                break

            self.draw_car()
            self.draw_obstacles()

            cv2.imshow("camera", frame)
            cv2.imshow("final", original)
            cv2.imshow("Car Dodging Game", self.game_window)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            else:
                self.handle_key_input(key)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = CarDodgingGame()
    game.run()

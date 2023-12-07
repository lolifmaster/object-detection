import cv2
import numpy as np
from cv1 import detection, tools
import random

# Initialize the game window
width, height = 400, 600
game_window = np.zeros((height, width, 3), dtype=np.uint8)

# Initialize car parameters
car_width, car_height = 50, 30
car_y = height - car_height - 10
car_x = width // 2 - car_width // 2

# Initialize obstacle parameters
obstacle_width, obstacle_height = 30, 30
obstacle_speed = 8
obstacles = []

# bounds 
lower_bound = np.array([90, 20, 90])
upper_bound = np.array([101, 38, 95])


# Function to draw the car
def draw_car(current_x):
    cv2.rectangle(game_window, pt1=(current_x, car_y), pt2=(current_x + car_width, car_y + car_height), color=(0, 255, 0))


# Function to draw obstacles
def draw_obstacles():
    for obstacle in obstacles:
        cv2.rectangle(game_window, pt1=(obstacle[0], obstacle[1]),
                      pt2=(obstacle[0] + obstacle_width, obstacle[1] + obstacle_height), color=(0, 0, 255))


# Function to move obstacles
def move_obstacles():
    global obstacles
    obstacles = [(x, y + obstacle_speed) for x, y in obstacles if y + obstacle_speed < height]


# Function to check collision with obstacles
def check_collision(current_x):
    for obstacle in obstacles:
        if (
                current_x < obstacle[0] + obstacle_width
                and current_x + car_width > obstacle[0]
                and car_y < obstacle[1] + obstacle_height
                and car_y + car_height > obstacle[1]
        ):
            return True
    return False


def generate_obstacle():
    obstacle_x = random.randint(0, width - obstacle_width)
    obstacle_y = -obstacle_height
    obstacles.append((obstacle_x, obstacle_y))


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

    hsv_frame = tools.bgr2hsv(frame)

    color_mask, contours = detection.in_range_detect(hsv_frame, lower_bound, upper_bound)

    original = tools.bitwise_and(frame, mask=color_mask)
    final = detection.draw_contours(original, contours, color=(255, 0, 0))

    center_x, _ = detection.calculate_center(contours)

    if center_x is not None:
        car_x = center_x

    if random.randint(0, 100) < 10:
        generate_obstacle()

    move_obstacles()

    if check_collision(car_x):
        print("Game Over!")
        break

    game_window[:] = 0
    draw_car(car_x)
    draw_obstacles()

    cv2.imshow("camera", frame)
    cv2.imshow("final", final)
    cv2.imshow("Car Dodging Game", game_window)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import random
from cv1 import tools, detection
from game_utils import Taxi, Police, Camion , rgba2rgb
import time 
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
    def __init__(self, width=400, height=600, car_width=50, car_height=80, obstacle_width=40, obstacle_height=30,
                 obstacle_speed=2, lower_bound=np.array([100, 50, 50]), upper_bound=np.array([130, 255, 255]), step: int = 20,
                 use_camera : bool = True):
        
        self.menu_window = None
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

        self.STEP = step 
        self.car_x = width // 2 - car_width // 2
        self.car_y = height - car_height - self.STEP -50

        self.obstacles = []

        self.use_camera = use_camera

        self.background = cv2.imread(r'assets/tunnel_road.jpg')
        self.background = cv2.resize(self.background, (width, height))

        self.car_image = cv2.imread(r'assets/mc_car.png', cv2.IMREAD_UNCHANGED)
        self.car_image = cv2.resize(self.car_image, (car_width, car_height))

        self.car_image=rgba2rgb(self.car_image)

        self.taxi = Taxi()
        self.camion = Camion()
        self.police = Police()

        self.start_time = time.time()
        self.current_time = 0
        self.acceleration_factor = 0.001
        self.spawn_probability = 0.001

        self.score = 0  

    def draw_car(self):
        """
        Draw the car on the game window.
        """

        # calculate the y-coordinates of the top and bottom of the car
        y1, y2 = self.car_y, self.car_y + self.car_height

        # calculate the x-coordinates of the left and right of the car
        x1, x2 = self.car_x, self.car_x + self.car_width

        # clip coordinates to stay within the game window
        # if y1 < 0 or y2 > self.height or x1 < 0 or x2 > self.width: put the min max values of the window
        y1 = max(0, y1)
        y2 = min(self.height, y2)
        x1 = max(0, x1)
        x2 = min(self.width, x2)

        # ensure the dimensions match
        # if the car is not in the window do not draw it
        car_image_height, car_image_width, _ = self.car_image.shape
        if y2 - y1 != car_image_height or x2 - x1 != car_image_width:
            return

        # draw the car image on the game window
        self.game_window[y1:y2, x1:x2, :] = self.car_image


        
    def draw_obstacles(self):
        """
        Draw the obstacles on the game window.
        """

        for obstacle in self.obstacles:
            # extract obstacle information
            obstacle_x, obstacle_y, obstacle_width, obstacle_height, obstacle_type = obstacle

            # load obstacle image and resize
            obstacle_image = cv2.imread(obstacle_type.image, cv2.IMREAD_UNCHANGED)
            obstacle_image = cv2.resize(obstacle_image, (obstacle_width, obstacle_height))

            # convert RGBA to RGB
            obstacle_image = rgba2rgb(obstacle_image)


            # ensure the dimensions of the obstacle image match the specified region
            obstacle_image_height, obstacle_image_width, _ = obstacle_image.shape
            if obstacle_image_height != obstacle_height or obstacle_image_width != obstacle_width:
                continue

            # update the game_window at the correct location
            y1, y2 = int(obstacle_y), int(obstacle_y + obstacle_height)
            x1, x2 = int(obstacle_x), int(obstacle_x + obstacle_width)

            # clip coordinates to stay within the game window
            y1 = max(0, y1)
            y2 = min(self.height, y2)
            x1 = max(0, x1)
            x2 = min(self.width, x2)

            # ensure the dimensions match
            game_window_region_height, game_window_region_width, _ = self.game_window[y1:y2, x1:x2, :].shape
        
            # check if the dimensions still match (additional check)
            if game_window_region_height != obstacle_height or game_window_region_width != obstacle_width:
                continue

            self.game_window[y1:y2, x1:x2, :] = obstacle_image


    def move_obstacles(self):
        """
        Move the obstacles down the game window.
        """
        new_obstacles = []
        
        for obstacle in self.obstacles:
            # extract obstacle information
            obstacle_x, obstacle_y, obstacle_width, obstacle_height, obstacle_type = obstacle

            # move the obstacle downward based on the obstacle speed
            obstacle_y += self.obstacle_speed

            # check if the obstacle has passed the lower edge of the game window
            if obstacle_y > self.height:
                # increase the score when the obstacle passes the lower fence
                self.score += obstacle_type.score  

            else:
                # add the obstacle to the new_obstacles list if it is still within the game window
                new_obstacles.append((obstacle_x, obstacle_y, obstacle_width, obstacle_height, obstacle_type))

        self.obstacles = new_obstacles
        
    def check_collision(self):
        """
        Check if the car has collided with any of the obstacles.
        """

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
        """
        Generate a new obstacle at a random location.
        """
        if len(self.obstacles) < 3:
            
            # choose a random obstacle type
            obstacle_type = random.choice([self.taxi, self.camion, self.police])

            # assign the obstacle's width and height
            obstacle_width = obstacle_type.width
            obstacle_height = obstacle_type.height

            # generate a new obstacle at a random location along the width of the game window
            obstacle_x = random.randint(0, self.width - obstacle_width)
            obstacle_y = -obstacle_height
 
            # check for collisions with existing obstacles
            if not any(
                obstacle_x < x + w and x < obstacle_x + obstacle_width and
                obstacle_y < y + h and y < obstacle_y + obstacle_height
                for x, y, w, h, _ in self.obstacles
            ):
                #check free space for the car always available for the car to move 
                if not any( 
                    self.car_x < x + w and x < self.car_x + self.car_width and
                    self.car_y < y + h and y < self.car_y + self.car_height
                    for x, y, w, h, _ in self.obstacles
                ):
                    self.obstacles.append((obstacle_x, obstacle_y, obstacle_width, obstacle_height, obstacle_type))


    def handle_key_input(self, key):
        """
        Handle the key input
        """
        # adjust the car's position based on the key input
        if key == 81:  
            self.car_x = max(0, self.car_x - self.STEP)
        elif key == 83:  
            self.car_x = min(self.width - self.car_width, self.car_x + self.STEP)

    
    def restart(self):
        """
        Restart the game.
        """
        self.__init__(use_camera=self.use_camera)
        self.run()

    def start_countdown(self):
        """
        Start the countdown before the game starts.
        """
        for i in range(3, 0, -1):
            self.menu_window = self.background.copy()
            cv2.putText(self.menu_window, "Car Dodging Game", (self.width // 2 - 130, self.height // 2 - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(self.menu_window, f"Score: {self.score}", (self.width // 2 - 50, self.height // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(self.menu_window, f"Time: {int(self.current_time)}", (self.width // 2 - 50, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(self.menu_window, f"Starting in {i}...", (self.width // 2 - 60, self.height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("Car Dodging Game Menu", self.menu_window)
            cv2.waitKey(1000)


    def run(self):
        """
        Run the game.
        """

        while True:
            # show the start screen with a countdown when pressing 's'
            self.menu_window = self.background.copy()
            cv2.putText(self.menu_window, "Press 's' to start", (self.width // 2 - 120, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("Car Dodging Game Menu", self.menu_window)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                self.start_countdown()
                break

        cap = None
        # start the game with camera if enabled
        if self.use_camera :
            cap = cv2.VideoCapture(0) 
            desired_width = 300
            desired_height = 150

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
            if self.use_camera and not cap.isOpened():
                print("Error: Could not open camera.")
                return

        while True:
            if self.use_camera:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                frame = cv2.flip(frame, 1)

            # put the background on the game window
            self.game_window = self.background.copy()
            
            if self.use_camera:
                hsv_frame = tools.bgr2hsv(frame)

                color_mask, contours = detection.in_range_detect(hsv_frame, self.lower_bound, self.upper_bound)

                original = tools.bitwise_and(frame, mask=color_mask)

                if contours:
                    original = detection.draw_contours(original, contours, color=(255, 0, 0))

                    center_x, _ = detection.calculate_center(contours)

                    if center_x is not None:
                        self.car_x = center_x

            if random.random() < self.spawn_probability:
                self.generate_obstacle()

            # update game time
            self.current_time = time.time() - self.start_time

            # adjust obstacle speed and spawn probability based on game time
            self.obstacle_speed += self.current_time * self.acceleration_factor
            self.spawn_probability += self.current_time * 0.0001 

            self.move_obstacles()

            if self.check_collision():
                print("Game Over!")
                break

            self.draw_car()
            self.draw_obstacles()
            
            cv2.putText(self.game_window, f"Score: {self.score}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if self.use_camera:
                cv2.imshow("final", original)
                cv2.imshow("image",frame)
                cv2.imshow("Car Dodging Game", self.game_window)
            else:
                cv2.imshow("Car Dodging Game", self.game_window)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            else:
                self.handle_key_input(key)

        if cap is not None:
            cap.release()
        cv2.destroyWindow("Car Dodging Game")

        # update the menu window with the final score
        self.menu_window = self.background.copy()
        cv2.putText(self.menu_window, "Game Over!", (self.width // 2 - 50, self.height // 2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.menu_window, f"Score: {self.score}", (self.width // 2 - 50, self.height // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.menu_window, f"Time: {int(self.current_time)}", (self.width // 2 - 50, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.menu_window, "Press 'p' to play again", (self.width // 2 - 120, self.height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.menu_window, "Press 'q' to quit", (self.width // 2 - 80, self.height // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Car Dodging Game Menu", self.menu_window)

        # Wait for 'p' key to restart the game
        while True:
            key = cv2.waitKey(1)
            if key & 0xFF == ord("p"):
                self.restart()
            elif key & 0xFF == ord("q"):
                break



if __name__ == "__main__":
    use_camera = True  
    game = CarDodgingGame(use_camera=use_camera)
    game.run()
from deepbots.supervisor import RobotSupervisorEnv
from utilities import normalize_to_range
import cv2
from gym.spaces import Box, Discrete
import gym
import numpy as np
import math
from controller import Display
from imageai.Detection.Custom import CustomObjectDetection
from PIL import Image

import sys

class HumanFollowingRobotSupervisor(RobotSupervisorEnv):


    def __init__(self,in_train,with_yolo):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        """

        super().__init__()
        
        self.lidar = self.getDevice("Hokuyo URG-04LX-UG01")
        self.lidar_fov = self.lidar.getFov()
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        self.lidar_radius = self.lidar.getMaxRange()
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Set up gym spaces
        self.observation_space = Box(low= np.concatenate([np.array([-np.inf, -np.inf, -1.0, -1.0, 0.0]),np.array([0.0]*self.lidar_resolution)]),
                                     high= np.concatenate([np.array([np.inf, np.inf, 1.0, 1.0, 6.28]),np.array([np.inf]*self.lidar_resolution)]),
                                     dtype=np.float64)
        self.action_space = Box(low=np.array([-1.0 , -1.0]), high=np.array([1.0 , 1.0]), dtype=np.float64)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.target = self.getFromDef('target')
        self.np_random, _ = gym.utils.seeding.np_random()
        self.np_random.seed(100)
        self.YOLO = with_yolo
        self.train = in_train
        #camera
        self.camera = self.getDevice("camera")
        self.camera.enable(self.timestep)
        self.position_sensor_camera = self.getDevice("camera_angle_sensor")
        self.position_sensor_camera.enable(self.timestep)
        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()
        self.display = self.getDevice('display_camera')
        self.display_lidar = self.getDevice('display_lidar')
        self.display_lidar_height = self.display_lidar.getHeight()
        self.display_lidar_width = self.display_lidar.getWidth()
        self.dist_to_goal_reached = 0.65
        self.dist_to_obstacle_reached = 0.30
        self.box_detect = None
        self.steps = 0
        if self.YOLO:
            print("Usando YOLO para detección de persona Objetivo")
            self.detector_yolo = CustomObjectDetection()
            self.detector_yolo.setModelTypeAsTinyYOLOv3()
            self.detector_yolo.setModelPath("tiny-yolov3_person_mAP-0.72476_epoch-101.pt")
            self.detector_yolo.setJsonPath("person_tiny-yolov3_detection_config.json")
            self.detector_yolo.loadModel()
            
        self.wheels = [None for _ in range(2)]
        self.obstacles = [None for _ in range(3)]
        self.motor_camera = None
        self.setup_motors()

        # Set up misc
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.prev_dist_to_goal = None
        self.init = True
        
        
    def yolo_detection(self):
        frame = self.camera.getImage()
        image = np.frombuffer(frame, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
        image = image[:, :, :3]    
        #detections = self.detector_yolo.detectObjectsFromImage(input_image=image,output_image_path="detect.png", minimum_percentage_probability=30)
        detections = self.detector_yolo.detectObjectsFromImage(input_image=image,minimum_percentage_probability=30)
        max_percentage = 0.0
        for Object in detections:    
            if Object["percentage_probability"] > max_percentage:
                self.box_detect = Object["box_points"]
                max_percentage = Object["percentage_probability"]
        #print("percentage max: ",max_percentage)
        
        self.display.setColor(0xFFFFFF)
        self.display.setOpacity(0.3)
        ir = self.display.imageNew(frame, Display.BGRA, self.width, self.height)
        self.display.imagePaste(ir, 0, 0, False)
        
        if self.box_detect:
            x1,y1,x2,y2 = self.box_detect
            self.display.fillRectangle(x1, y1, x2 - x1, y2 - y1)
            self.box_detect = None
              
        self.display.imageDelete(ir)
        
    def saving_images(self):
        frame = self.camera.getImage()
        image = np.frombuffer(frame, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
        image = image[:, :, :3]
        cv2.imwrite("images/Person"+ str(self.steps) +".png", image)
        
    def lidar_show_display(self,range_image):
        dw = self.display_lidar_width
        dh = self.display_lidar_height
        fov = self.lidar_fov
        dw2 = dw // 2
        dh2 = dh // 2
        fov2 = fov / 2
        px = [dw2, dw2 + int(dw * math.cos(-fov2 - math.pi / 2)), dw2 + int(dw * math.cos(fov2 - math.pi / 2))]
        py = [dh2, dh2 + int(dh * math.sin(-fov2 - math.pi / 2)), dh2 + int(dh * math.sin(fov2 - math.pi / 2))]
        self.display_lidar.setColor(0x002642)
        self.display_lidar.fillRectangle(0,0,dw,dh)
        self.display_lidar.setColor(0xFFFFFF)
        self.display_lidar.fillPolygon(px,py)
        self.display_lidar.drawPolygon(px,py)
        self.display_lidar.drawLine(dw2, 0, dw2, dh)
        self.display_lidar.drawLine(0, dh2, dw, dh2)
        self.display_lidar.drawOval(dw2, dh2, dw2, dh2)
        self.display_lidar.setColor(0x92CEF7)
        
        for i in range(self.lidar_resolution):
            f = range_image[i]
            if f != float('inf') and not math.isnan(f):
                alpha = -fov2 + fov * i/self.lidar_resolution - math.pi/2
                self.display_lidar.drawLine(dw2, dh2, dw2 + f * math.cos(alpha) * dw2/self.lidar_radius, 
                                                      dh2 + f * math.sin(alpha) * dh2/self.lidar_radius)
        
    def get_observations(self):
        
        if self.YOLO and self.steps%4 == 0:
            self.yolo_detection()
        
        #self.saving_images()
        
        
        robot_velocity_x = normalize_to_range(self.robot.getVelocity()[0], -6.0, 6.0, -1.0, 1.0, clip=True)
        robot_velocity_y = normalize_to_range(self.robot.getVelocity()[1], -6.0, 6.0, -1.0, 1.0, clip=True)

        robot_orientation_x = normalize_to_range(self.robot.getOrientation()[0], -1.0, 1.0, -1.0, 1.0, clip=True)
        robot_orientation_y = normalize_to_range(self.robot.getOrientation()[1], -1.0, 1.0, -1.0, 1.0, clip=True)
          
        angle_camera = normalize_to_range(self.sensor_camera_value(), 0.0, 6.28, -1.0, 1.0, clip=True)
          
        range_image = self.lidar.getRangeImage()
        range_image = np.clip(range_image, 0, self.lidar_radius)
        self.lidar_show_display(range_image)
        range_image = np.interp(range_image, (0, self.lidar_radius), (0, 1))
        range_image = range_image.tolist()
        self.steps = self.steps + 1
        
        return [robot_velocity_x, robot_velocity_y, 
                robot_orientation_x, robot_orientation_y, 
                angle_camera] + range_image

    def get_reward(self, action):
        dist_to_goal = math.sqrt(((self.robot.getPosition()[0] - self.target.getPosition()[0]) ** 2 +
                                  (self.robot.getPosition()[1] - self.target.getPosition()[1]) ** 2))
        
        reward = max(self.prev_dist_to_goal - dist_to_goal,0)*10
        if self.prev_dist_to_goal - dist_to_goal < 0:
            reward = -0.001
        if dist_to_goal < self.dist_to_goal_reached:
            reward = 100.0
        self.prev_dist_to_goal = dist_to_goal
        return reward
        

    def is_done(self):
        if self.prev_dist_to_goal < self.dist_to_goal_reached and self.train:
            print("Goal")
            self.steps = 0
            return True
            
        #the radious of the robot is 0.265    
        for laser in self.lidar.getRangeImage():
            if laser < self.dist_to_obstacle_reached and self.train:
                print("Obstacle")
                self.steps = 0
                return True
                
        if abs(self.robot.getPosition()[0]) > 4.0 or abs(self.robot.getPosition()[1]) > 4.0:
            print("Arena Limit")
            return True
                
        return False

    def solved(self):

        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 125.0:  # Last 100 episode scores average value
                return True
        return False

    def aleatory_position(self,radio_min ,radio_max,angle):
        radio = np.random.uniform(radio_min, radio_max)
        x = radio * np.cos(angle)
        y = radio * np.sin(angle)
        return x,y
        
    def setup_obstacles(self):
        random_angles = [None] * 6
        # Set place for each obstacle:
        for i in range(3):
            self.obstacles[i] = self.getFromDef('obstacle'+str(i+1))
            random_angles[i] = np.random.uniform(0, 2*np.pi)
            x,y = self.aleatory_position(radio_min = 1.9, radio_max = 1.9,angle = random_angles[i])
            trans_field = self.obstacles[i].getField("translation")
            trans_field.setSFVec3f([x,y,0.40])
            
        random_angles[3] = np.random.uniform(0, 2*np.pi)
        random_angles[4] = np.random.uniform(0, 2*np.pi)
        random_angles[5] = np.random.uniform(0, 2*np.pi)
        # Set place of target behind random obstacle:
        x_target,y_target = self.aleatory_position(radio_min = 3.0, radio_max = 3.0,
                                                    angle = random_angles[np.random.randint(0, 5)])
        self.prev_dist_to_goal = math.sqrt(((self.robot.getPosition()[0] - x_target) ** 2 +
                                  (self.robot.getPosition()[1] - y_target) ** 2))
        trans_field = self.target.getField("translation")
        trans_field.setSFVec3f([x_target,y_target,0.06])

    def get_default_observation(self):
        # When reset is activated
        self.setup_obstacles()
        return np.concatenate([np.array([0.0,0.0,1.0,0.0,0.0]),np.array([1.0]*self.lidar_resolution)])
        
    def change_direction_camera(self):
        # angle_robot -> [0,6.2830]
        angle_robot = 3.1415 + math.atan2(self.robot.getOrientation()[1],self.robot.getOrientation()[0])
        # angle_persona -> [0,6.2830]
        angle_persona = 3.1415 - math.atan2(-self.robot.getPosition()[1] + self.target.getPosition()[1],
                            -self.robot.getPosition()[0] + self.target.getPosition()[0]) 

        angle_camera = angle_persona - angle_robot 
        angle_camera = angle_camera if angle_camera>0 else 6.2830+angle_camera  
        
        
        #print(self.sensor_camera_value())
        
        if self.sensor_camera_value() < angle_camera:
            if abs(self.sensor_camera_value() - angle_camera)<3.1415:
                if abs(self.sensor_camera_value()-angle_camera) < 0.01 :
                    self.motor_camera.setPosition(float('inf'))
                    self.motor_camera.setVelocity(0.0)
                else:
                    self.motor_camera.setPosition(float('inf'))
                    self.motor_camera.setVelocity(-1.5)
                #print("1")
            else:
                self.motor_camera.setPosition(float('inf'))
                self.motor_camera.setVelocity(1.5)
                #print("2")
                
        if self.sensor_camera_value() > angle_camera:
            if abs(self.sensor_camera_value() - angle_camera)<3.1415:
                if abs(self.sensor_camera_value()-angle_camera) < 0.01 :
                    self.motor_camera.setPosition(float('inf'))
                    self.motor_camera.setVelocity(0.0)
                else:
                    self.motor_camera.setPosition(float('inf'))
                    self.motor_camera.setVelocity(1.5)
                #print("3")
            else:
                self.motor_camera.setPosition(float('inf'))
                self.motor_camera.setVelocity(-1.5)
                #print("4")
            
    def sensor_camera_value(self):
        if self.position_sensor_camera.getValue()<0:
            return 6.28 - self.position_sensor_camera.getValue()%6.2830 
        else:
            return -self.position_sensor_camera.getValue()%6.2830  
        return value

    def apply_action(self, action):

        speed_right = float(action[0]) * 4.0  # 
        speed_left = float(action[1]) * 4.0
        #print(action[0],action[1])
 
        self.wheels[0].setPosition(float('inf'))
        self.wheels[0].setVelocity(speed_right)
        self.wheels[1].setPosition(float('inf'))
        self.wheels[1].setVelocity(speed_left)
        #self.motor_camera.setPosition(float('inf'))
        self.change_direction_camera()
        
        #print(self.position_sensor_camera.getValue())

    def setup_motors(self):
        """
        This method initializes the four wheels, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.wheels[0] = self.getDevice('wheel_right_joint')
        self.wheels[1] = self.getDevice('wheel_left_joint')
        self.motor_camera = self.getDevice('camera_motor')
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0.0)
        self.motor_camera.setPosition(float('inf'))
        self.motor_camera.setVelocity(0.0)

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render
        :param mode:
        :return:
        """
        print("render() is not used")

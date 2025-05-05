#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import time
import threading  # 
import sys  # 
import select  # 
import termios  # 
import tty  # 

# Constants
TURNING_SPEED = 0.3 / 100
MOVING_SPEED = 0.1
TURN_SPEED = 0.5  # Speed for turning away from walls
CENTER_TOLERANCE = 20  # Error margin for "centered" opponent
WALL_DISTANCE = 0.25  # Minimum distance from wall (meters) (4 in ~ 0.1 m)
ATTACK_SPEED = 0.6  # Increased attack speed
MIN_CONTOUR_SIZE = 200 
RED_UPPER = [92, 255, 155] 
RED_LOWER = [0, 149, 124]
GREEN_UPPER = [255, 117, 255]
GREEN_LOWER = [47, 0, 135]
PURPLE_UPPER = [86, 176, 138]
PURPLE_LOWER = [0, 148, 104]

TEAM_COLOR = "purple"
GOAL_UPPER = PURPLE_UPPER if TEAM_COLOR == "purple" else GREEN_UPPER
GOAL_LOWER = PURPLE_LOWER if TEAM_COLOR == "purple" else GREEN_LOWER

CENTER_LIDAR_TOLERANCE = 0.05

class ColorTracking(Node):
    def __init__(self, name):
        super().__init__(name)

        # ROS2 publishers and subscribers
        self.cmd_vel = self.create_publisher(Twist, 'controller/cmd_vel', 1)

        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan_raw', self.lidar_callback, 1)
        
        self.start = True
        
        self.subscription = self.create_subscription(
            Image,
            'ascamera/camera_publisher/rgb0/image',
            self.listener_callback,
            1)
        self.start = False
        self.charge = False
        self.bridge = CvBridge()
        self.state = "charge"

    def lidar_callback(self, data):
        """Process Lidar data for wall detection and navigation"""
        if not self.active:  # 
            return  # 
        self.lidar_data = data.ranges 

        if self.state == "center":
            self.move_to_center()
        
    def move_to_center(self):
        if not self.lidar_data or len(self.lidar_data) < 360:
            return []

        angle_increment = 2 * np.pi / len(self.lidar_data)
        direction_angles = {
            "E": -np.pi / 2, #right
            "N": 0, #forward
            "W": np.pi / 2, #left
            "S": np.pi #backward
        }

        right_distance = 0.0
        left_distance = 0.0
        front_distance = 0.0
        back_distance = 0.0

        for direction, angle in direction_angles.items():
            range_deg = 15
            range_rad = np.deg2rad(range_deg)
            range_indices = int(range_rad / angle_increment)
            center_index = int((angle % (2 * np.pi)) / angle_increment)
            indices = [(center_index + i) % len(self.lidar_data) for i in range(-range_indices, range_indices + 1)]
            distances = [self.lidar_data[i] for i in indices if not np.isnan(self.lidar_data[i])]
            if distances:
                if direction == "E":
                    right_distance = min(distances)
                elif direction == "N":
                    front_distance = min(distances)
                elif direction == "S":
                    back_distance = min(distances)
                elif direction == "W":
                    left_distance = min(distances)

        in_center = True

        twist = Twist()
        
        if abs(left_distance - right_distance) > CENTER_LIDAR_TOLERANCE:
            in_center = False
            if left_distance > right_distance:
                twist.linear.y = -0.2
            else:
                twist.linear.y = 0.2

        if abs(front_distance - back_distance) > CENTER_LIDAR_TOLERANCE:
            in_center = False
            if front_distance > back_distance:
                twist.linear.x = 0.2
            else:
                twist.linear.x = -0.2

        self.cmd_vel.publish(twist)

        if in_center:
            self.state = "return"

    def listener_callback(self, data):
        """ Process camera input and decide movement. """
        if not self.active:  # 
            return  # 
        current_frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Convert BGR to LAB color space
        lab_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2LAB)
        blurred_frame = cv2.GaussianBlur(lab_frame, (5, 5), 0)

        # Define LAB color range for detecting red
        lower_bound = np.array(RED_LOWER)
        upper_bound = np.array(RED_UPPER)

        # Create a binary mask for detected color
        mask = cv2.inRange(blurred_frame, lower_bound, upper_bound)

        # Get centroid of detected object
        centroid_x, centroid_y = self.get_color_centroid(mask)

        twist = Twist()

        # Check if we can charge red
        if centroid_x is not None and centroid_y is not None:
            # Target detected: Align and move forward
            image_center_x = current_frame.shape[1] // 2
            error_x = centroid_x - image_center_x
            
            if abs(error_x) < CENTER_TOLERANCE:
                self.charge = True
           
            twist.angular.z = -error_x * TURNING_SPEED  # Proportional turn
            self.state = "charge"
            twist.linear.x = ATTACK_SPEED if self.charge else 0.0
            self.cmd_vel.publish(twist)
        elif self.state == "charge":
            self.state = "center"
            self.charge = False
        elif self.state == "return":
            # return to goal

            # check to see if we're at the goal
            if min(self.lidar_data) < WALL_DISTANCE:
                self.state = "spin"
                return

            # find goal color
            returning = False
            lower_bound = np.array(GOAL_LOWER)
            upper_bound = np.array(GOAL_UPPER)

            # Create a binary mask for detected color
            mask = cv2.inRange(blurred_frame, lower_bound, upper_bound)

            # Get centroid of detected object
            centroid_x, centroid_y = self.get_color_centroid(mask)

            if centroid_x is not None and centroid_y is not None:
                # Target detected: Align and move forward
                image_center_x = current_frame.shape[1] // 2
                error_x = centroid_x - image_center_x
                
                if abs(error_x) < CENTER_TOLERANCE:
                    returning = True
                
                twist.angular.z = -error_x * TURNING_SPEED  # Proportional turn
            else:
                # twist to find goal
                twist.angular.z = 1.25
                self.state = "charge"

            twist.linear.x = MOVING_SPEED if returning else 0.0

            self.cmd_vel.publish(twist)
        elif self.state == "spin":
            # No target detected: Rotate to scan for ball
            twist.angular.z = 1.25
            self.charge = False
            twist.linear.x = 0.0
            self.cmd_vel.publish(twist)
        

    def get_color_centroid(self, mask):
        """ Compute the centroid of the largest detected contour. """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < MIN_CONTOUR_SIZE:
            return None, None

        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, None

        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        return centroid_x, centroid_y
        
def keyboard_listener(node):  # 
    print("Press ENTER to start, 'p' to pause, 'q' to quit.")  # 
    tty.setcbreak(sys.stdin.fileno())  # 
    while not node.shutdown_requested:  # 
        if select.select([sys.stdin], [], [], 0.1)[0]:  # 
            ch = sys.stdin.read(1)  # 
            if ch == '\n':  # 
                print("Starting...")  # 
                node.active = True  # 
            elif ch == 'p':  # 
                print("Paused.")  # 
                node.active = False  # 
            elif ch == 'q':  # 
                print("Quitting...")  # 
                node.shutdown_requested = True  # 
                rclpy.shutdown()  # 
                
def main(args=None):
    rclpy.init(args=args)
    color_tracking_node = ColorTracking('color_tracking_node')

    listener_thread = threading.Thread(target=keyboard_listener, args=(color_tracking_node,), daemon=True)  # 
    listener_thread.start()  # 
    
    try:
        while rclpy.ok() and not color_tracking_node.shutdown_requested:  # 
            rclpy.spin_once(color_tracking_node, timeout_sec=0.1)  # 
    except KeyboardInterrupt:
        color_tracking_node.get_logger().info("Keyboard Interrupt (Ctrl+C): Stopping node.")
    finally:
        color_tracking_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

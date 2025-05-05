#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import time

class MazeNavigator(Node):
    def __init__(self):
        super().__init__('maze_navigator')

        # Publisher for robot velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'controller/cmd_vel', 10)

        # Subscribers for camera and lidar
        self.camera_sub = self.create_subscription(
            Image, 'ascamera/camera_publisher/rgb0/image',
            self.camera_callback, 10)

        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan_raw', self.lidar_callback, 10)

        self.bridge = CvBridge()

        # Latest lidar ranges
        self.lidar_data = []

        # Movement constants
        self.FORWARD_SPEED = 0.2
        self.BACKWARD_SPEED = 0.15
        self.TURN_SPEED = 0.3
        self.END_DISTANCE = 0.10  # meters, desired distance from wall
        self.CENTER_TOLERANCE = 0.02  # meters tolerance for centering

        # State Flags
        self.state = 'searching'  # states: searching, approaching_ball, moving_back, centering, stopped

        # Timer to run control loop at 10Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def camera_callback(self, msg):
        # Convert ROS Image to OpenCV BGR
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert BGR to HSV for color detection
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red color range in HSV (two parts for wrapping hue)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 150:  # minimal area threshold for filtering noise
                # Found a red ball
                if self.state == 'searching':
                    self.get_logger().info('Red ball detected, approaching...')
                    self.state = 'approaching_ball'
                # Could add logic here for ball location for driving direction
                return

        # If no ball detected and currently approaching, revert to search mode
        if self.state == 'approaching_ball':
            self.get_logger().info('Lost red ball, searching again.')
            self.state = 'searching'

    def lidar_callback(self, msg):
        self.lidar_data = msg.ranges

    def control_loop(self):
        # Main control loop called periodically
        if not self.lidar_data or len(self.lidar_data) < 360:
            # No lidar data yet
            self.stop_motion()
            return

        twist = Twist()

        if self.state == 'searching':
            # Rotate slowly to search for red ball
            twist.angular.z = 0.3
            twist.linear.x = 0.0

        elif self.state == 'approaching_ball':
            # Move forward to hit the ball
            twist.linear.x = self.FORWARD_SPEED
            twist.angular.z = 0.0

            # Check front lidar distance to stop moving forward if obstacle close
            front_distance = self.lidar_data[180]
            if front_distance < 0.15:
                self.get_logger().info('Hit the ball, moving backward now.')
                self.state = 'moving_back'

        elif self.state == 'moving_back':
            front_distance = self.lidar_data[180]
            if front_distance > self.END_DISTANCE:
                # Move backward until close enough to wall
                twist.linear.x = -self.BACKWARD_SPEED
                twist.angular.z = 0.0
            else:
                self.get_logger().info('Reached target distance from wall, centering...')
                self.state = 'centering'
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        elif self.state == 'centering':
            # Get distances on left and right to center
            left_distance = self.lidar_data[90]
            right_distance = self.lidar_data[270]

            if left_distance == 0 or right_distance == 0:
                # Invalid distance readings, stop and wait
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                diff = left_distance - right_distance

                if abs(diff) < self.CENTER_TOLERANCE:
                    # Centered enough, stop
                    self.get_logger().info('Centered between walls. Stopping.')
                    self.state = 'stopped'
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                else:
                    # Turn proportionally to difference
                    turn_direction = -1.0 if diff > 0 else 1.0
                    twist.angular.z = turn_direction * self.TURN_SPEED
                    twist.linear.x = 0.0

        elif self.state == 'stopped':
            # Stay stopped
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        else:
            # Default fallback: stop motion
            self.stop_motion()
            return

        self.cmd_vel_pub.publish(twist)

    def stop_motion(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)

    navigator = MazeNavigator()
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info('Keyboard Interrupt (Ctrl+C): Stopping node.')
    finally:
        navigator.stop_motion()
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.kp = 0.5
        self.ki = 0.0
        self.kd = 0.0

        self.integral   = 0.0
        self.prev_error = 0.0
        self.prev_time  = self.get_clock().now()
        self.cb_count   = 0

        self.desired_dist = 1.0  # meters from left wall
        self.lookahead = 1.4     # L
        self.theta = np.deg2rad(45.0)

    def get_range(self, range_data, angle, angle_min, angle_increment):
        idx = int(round((angle - angle_min) / angle_increment))
        idx = np.clip(idx, 0, len(range_data) - 1)
        r = range_data[idx]
        if np.isnan(r) or np.isinf(r):
            return 0.0
        return float(r)

    def get_error(self, range_data, angle_min, angle_increment):
        b = self.get_range(range_data, np.pi / 2.0,              angle_min, angle_increment)
        a = self.get_range(range_data, np.pi / 2.0 - self.theta, angle_min, angle_increment)

        alpha = np.arctan2(a * np.cos(self.theta) - b, a * np.sin(self.theta))
        Dt = b * np.cos(alpha)
        Dt1 = Dt + self.lookahead * np.sin(alpha)

        return Dt1 - self.desired_dist

    def pid_control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        angle = self.kp * error + self.ki * self.integral + self.kd * derivative
        angle = float(np.clip(angle, -np.deg2rad(24.0), np.deg2rad(24.0)))

        abs_deg = abs(np.rad2deg(angle))
        if abs_deg <= 10.0:
            speed = 1.5
        elif abs_deg <= 20.0:
            speed = 1.0
        else:
            speed = 0.5

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = angle
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

        return angle, speed

    def scan_callback(self, scan_msg):
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds * 1e-9
        self.prev_time = now

        ranges = np.array(scan_msg.ranges, dtype=np.float64)
        error = self.get_error(ranges, scan_msg.angle_min, scan_msg.angle_increment)
        angle, speed = self.pid_control(error, dt)



def main(args=None):
    rclpy.init(args=args)
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

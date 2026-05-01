#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool


class DriveMuxNode(Node):
    def __init__(self):
        super().__init__('drive_mux')

        self.declare_parameter('mode', 'pp')  # 'pp' | 'gap' | 'off'
        self.declare_parameter('auto_mode', True)

        self.declare_parameter('auto_obstacle_topic', '/obstacle_ahead')
        self.declare_parameter('auto_obstacle_timeout', 0.15)
        self.declare_parameter('auto_obstacle_hold', 0.3)

        self.declare_parameter('timeout', 0.2)
        self.declare_parameter('output_topic', '/drive')
        self.declare_parameter('pp_topic', '/drive_pp')
        self.declare_parameter('gap_topic', '/drive_gap')
        self.declare_parameter('publish_rate', 20.0)

        self.mode = str(self.get_parameter('mode').value)
        self.auto_mode = bool(self.get_parameter('auto_mode').value)
        self.auto_obstacle_topic = str(self.get_parameter('auto_obstacle_topic').value)
        self.auto_obstacle_timeout = float(self.get_parameter('auto_obstacle_timeout').value)
        self.auto_obstacle_hold = float(self.get_parameter('auto_obstacle_hold').value)
        self.timeout = float(self.get_parameter('timeout').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.pp_topic = str(self.get_parameter('pp_topic').value)
        self.gap_topic = str(self.get_parameter('gap_topic').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        qos = QoSProfile(depth=10)
        self.pub = self.create_publisher(AckermannDriveStamped, self.output_topic, 10)
        self.sub_pp = self.create_subscription(AckermannDriveStamped, self.pp_topic, self._on_pp, qos)
        self.sub_gap = self.create_subscription(AckermannDriveStamped, self.gap_topic, self._on_gap, qos)
        self.sub_obstacle = self.create_subscription(Bool, self.auto_obstacle_topic, self._on_obstacle, qos)

        self.last_pp = None
        self.last_gap = None
        self.last_pp_time = None
        self.last_gap_time = None
        self.last_obstacle = None
        self.last_obstacle_time = None
        self.last_obstacle_trigger = None
        self.last_mode = None

        self.timer = self.create_timer(1.0 / max(self.publish_rate, 1.0), self._publish)

        self.get_logger().info(
            f"drive_mux up: mode='{self.mode}', auto_mode={self.auto_mode}, output='{self.output_topic}', "
            f"pp='{self.pp_topic}', gap='{self.gap_topic}', timeout={self.timeout:.2f}s"
        )

    def _on_pp(self, msg: AckermannDriveStamped):
        self.last_pp = msg
        self.last_pp_time = self.get_clock().now()

    def _on_gap(self, msg: AckermannDriveStamped):
        self.last_gap = msg
        self.last_gap_time = self.get_clock().now()

    def _on_obstacle(self, msg: Bool):
        self.last_obstacle = bool(msg.data)
        self.last_obstacle_time = self.get_clock().now()
        if self.last_obstacle:
            self.last_obstacle_trigger = self.last_obstacle_time

    def _is_fresh(self, stamp):
        if stamp is None:
            return False
        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
        return age <= self.timeout

    def _obstacle_fresh(self):
        if self.last_obstacle_time is None:
            return False
        age = (self.get_clock().now() - self.last_obstacle_time).nanoseconds * 1e-9
        return age <= self.auto_obstacle_timeout

    def _resolve_mode(self):
        auto_mode = bool(self.get_parameter('auto_mode').value)
        if not auto_mode:
            return str(self.get_parameter('mode').value)

        auto_pp = 'pp'
        auto_fallback = 'gap'

        hold = float(self.get_parameter('auto_obstacle_hold').value)
        now = self.get_clock().now()

        if self.last_obstacle is True and self._obstacle_fresh():
            if self.last_obstacle_trigger is None:
                self.last_obstacle_trigger = now
            return auto_fallback

        if self.last_obstacle_trigger is not None:
            age = (now - self.last_obstacle_trigger).nanoseconds * 1e-9
            if age < hold:
                return auto_fallback

        return auto_pp

    def _publish(self):
        mode = self._resolve_mode()
        self.mode = mode

        if mode != self.last_mode:
            self.get_logger().info(
                f"mode switch: '{self.last_mode}' -> '{mode}', obstacle={self.last_obstacle}"
            )
            self.last_mode = mode

        if mode == 'gap':
            if self.last_gap is not None and self._is_fresh(self.last_gap_time):
                self.pub.publish(self.last_gap)
            elif self.last_pp is not None and self._is_fresh(self.last_pp_time):
                self.pub.publish(self.last_pp)
            else:
                self._publish_stop()
            return

            # if self.last_pp is not None and self._is_fresh(self.last_pp_time):
            #     self.pub.publish(self.last_pp)
            # else:
            #     self._publish_stop()
            # return

        if mode == 'pp':
            if self.last_pp is not None and self._is_fresh(self.last_pp_time):
                self.pub.publish(self.last_pp)
            else:
                self._publish_stop()
            return

        self._publish_stop()

    def _publish_stop(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = 0.0
        msg.drive.speed = 0.0
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DriveMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

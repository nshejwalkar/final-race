#!/usr/bin/env python3
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


# =====================================================================
# USER-TUNABLE PARAMETERS
# =====================================================================
@dataclass
class ControllerConfig:
    # ---------------- Topics / frames ----------------
    pose_topic: str = '/pf/pose/odom'     # real-car localization source
    # pose_topic: str = '/ego_racecar/odom'   # sim
    drive_topic: str = '/drive_pp'
    marker_topic: str = '/pure_pursuit/markers'
    global_frame: str = 'map'
    publish_markers: bool = True

    # ---------------- Waypoint file ----------------
    waypoints_path: str = ''               # overridden at launch via ROS param or main()
    loop: bool = True

    # ---------------- Path preprocessing ----------------
    resample_spacing: float = 0.20
    smooth_window_size: int = 1
    dedup_distance: float = 0.05
    max_reacquire_dist: float = 5.50
    nearest_search_window: int = 80

    # ---------------- Pure pursuit geometry ----------------
    wheelbase: float = 0.33
    lookahead_base: float = 0.85
    lookahead_gain: float = 0.25
    min_lookahead: float = 0.65
    max_lookahead: float = 1.60
    target_lateral_offset: float = 0.0

    # ---------------- Steering limits ----------------
    steer_gain: float = 1.00
    max_steer: float = 0.4189
    max_steer_rate: float = 1.20

    # ---------------- Speed logic ----------------
    use_waypoint_velocity: bool = True
    nominal_speed: float = 2.50
    min_speed: float = 0.60
    max_speed: float = 4.00
    curve_slowdown_gain: float = 1.50
    curve_speed_power: float = 1.00
    max_accel: float = 1.20
    max_decel: float = 2.50

    # ---------------- Timing / safety ----------------
    control_rate: float = 20.0
    pose_timeout: float = 0.30
    stop_on_path_end: bool = False



CFG = ControllerConfig()


class PurePursuitNode(Node):
    """
    Pure pursuit controller. Waypoint CSV columns: x, y[, v[, l]]
    """

    def __init__(self, cfg: ControllerConfig):
        super().__init__('pure_pursuit_node')
        self.cfg = cfg

        # Allow key config to be overridden from a YAML param file or launch arg
        self.declare_parameter('waypoints_path', cfg.waypoints_path)
        self.declare_parameter('pose_topic', cfg.pose_topic)

        wp = self.get_parameter('waypoints_path').value
        if wp:
            self.cfg.waypoints_path = wp
        pt = self.get_parameter('pose_topic').value
        if pt:
            self.cfg.pose_topic = pt

        self.validate_parameters()

        # ---------------- State ----------------
        self.current_speed = 0.0
        self.current_pose = None
        self.last_pose_stamp = None
        self.last_steer_time = None
        self.last_speed_time = None
        self.last_steer_cmd = 0.0
        self.last_speed_cmd = 0.0
        self.nearest_idx = 0
        self.last_goal_idx = 0
        self.warned_stale = False
        self.warned_lost = False


        # ---------------- Path ----------------
        self.waypoints_xy, self.waypoint_speeds, self.waypoint_lookaheads = \
            self.load_and_prepare_waypoints(self.cfg.waypoints_path)
        self.num_waypoints = int(self.waypoints_xy.shape[0])
        self.arc_lengths = self.compute_arc_lengths(self.waypoints_xy, cfg.loop)
        self.track_length = float(self.arc_lengths[-1]) if self.arc_lengths.size > 0 else 0.0

        self.get_logger().info(
            f'Loaded {self.num_waypoints} waypoints, track length {self.track_length:.2f} m.'
        )
        self.get_logger().info(f'Pose source: {cfg.pose_topic}')
        self.get_logger().info(f'Waypoints: {self.cfg.waypoints_path}')

        # ---------------- ROS I/O ----------------
        self.pose_sub = self.create_subscription(
            Odometry, cfg.pose_topic, self.odom_callback, qos_profile_sensor_data,
        )
        self.drive_pub = self.create_publisher(AckermannDriveStamped, cfg.drive_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, cfg.marker_topic, 10)
        self.control_timer = self.create_timer(1.0 / cfg.control_rate, self.control_loop)

    # ==================================================================
    # Parameter validation
    # ==================================================================
    def validate_parameters(self):
        c = self.cfg

        def require(cond, msg):
            if not cond:
                raise ValueError(msg)

        require(c.waypoints_path.strip() != '', 'waypoints_path must not be empty.')
        require(c.resample_spacing > 0.01, 'resample_spacing must be > 0.01 m.')
        require(c.dedup_distance >= 0.0, 'dedup_distance must be >= 0.')
        require(c.max_reacquire_dist > 0.0, 'max_reacquire_dist must be > 0.')
        require(c.nearest_search_window >= 5, 'nearest_search_window must be >= 5.')
        require(c.wheelbase > 0.0, 'wheelbase must be > 0.')
        require(c.min_lookahead > 0.0, 'min_lookahead must be > 0.')
        require(c.max_lookahead >= c.min_lookahead, 'max_lookahead must be >= min_lookahead.')
        require(c.max_steer > 0.0, 'max_steer must be > 0.')
        require(c.max_steer_rate > 0.0, 'max_steer_rate must be > 0.')
        require(c.control_rate > 1.0, 'control_rate must be > 1 Hz.')
        require(c.pose_timeout > 0.0, 'pose_timeout must be > 0.')
        require(c.min_speed >= 0.0, 'min_speed must be >= 0.')
        require(c.max_speed >= c.min_speed, 'max_speed must be >= min_speed.')
        require(c.nominal_speed >= 0.0, 'nominal_speed must be >= 0.')
        require(c.max_accel > 0.0, 'max_accel must be > 0.')
        require(c.max_decel > 0.0, 'max_decel must be > 0.')
        require(c.curve_slowdown_gain >= 0.0, 'curve_slowdown_gain must be >= 0.')
        require(c.curve_speed_power > 0.0, 'curve_speed_power must be > 0.')

    # ==================================================================
    # Waypoint loading / preprocessing
    # ==================================================================
    def load_and_prepare_waypoints(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_xy, raw_v, raw_l = self.read_waypoint_csv(csv_path)
        clean_xy, clean_v, clean_l = self.remove_duplicate_waypoints(
            raw_xy, raw_v, raw_l, self.cfg.dedup_distance
        )

        min_points = 3 if self.cfg.loop else 2
        if clean_xy.shape[0] < min_points:
            raise ValueError(
                f'Waypoint file has only {clean_xy.shape[0]} valid unique points; need {min_points}.'
            )

        proc_xy, proc_v, proc_l = self.resample_waypoints(
            clean_xy, clean_v, clean_l, self.cfg.resample_spacing, self.cfg.loop,
        )

        if self.cfg.smooth_window_size > 1:
            proc_xy = self.smooth_path(proc_xy, self.cfg.smooth_window_size, self.cfg.loop)

        proc_v = np.clip(proc_v, self.cfg.min_speed, self.cfg.max_speed)
        proc_l = np.clip(proc_l, self.cfg.min_lookahead, self.cfg.max_lookahead)

        return proc_xy, proc_v, proc_l

    def read_waypoint_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Waypoint CSV not found: {path}')

        points, speeds, lookaheads = [], [], []
        with path.open('r', newline='') as f:
            for row in csv.reader(f):
                if not row:
                    continue
                row = [item.strip() for item in row]
                if len(row) < 2 or row[0].startswith('#'):
                    continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                    v = float(row[2]) if len(row) >= 3 and row[2] != '' else self.cfg.nominal_speed
                    l = float(row[3]) if len(row) >= 4 and row[3] != '' else self.cfg.lookahead_base
                except ValueError:
                    continue  # skip header rows
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                if not np.isfinite(v):
                    v = self.cfg.nominal_speed
                if not np.isfinite(l):
                    l = self.cfg.lookahead_base
                points.append((x, y))
                speeds.append(v)
                lookaheads.append(l)

        if not points:
            raise ValueError(f'No valid waypoint rows found in: {path}')

        return (
            np.asarray(points, dtype=np.float64),
            np.asarray(speeds, dtype=np.float64),
            np.asarray(lookaheads, dtype=np.float64),
        )

    @staticmethod
    def remove_duplicate_waypoints(xy, v, l, min_dist):
        if xy.shape[0] <= 1 or min_dist <= 0.0:
            return xy, v, l
        kept_xy, kept_v, kept_l = [xy[0]], [v[0]], [l[0]]
        for i in range(1, xy.shape[0]):
            if np.linalg.norm(xy[i] - kept_xy[-1]) >= min_dist:
                kept_xy.append(xy[i])
                kept_v.append(v[i])
                kept_l.append(l[i])
        return (
            np.asarray(kept_xy, dtype=np.float64),
            np.asarray(kept_v, dtype=np.float64),
            np.asarray(kept_l, dtype=np.float64),
        )

    @staticmethod
    def compute_arc_lengths(xy: np.ndarray, loop: bool) -> np.ndarray:
        if xy.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        if loop:
            segs = np.linalg.norm(np.diff(np.vstack([xy, xy[0]]), axis=0), axis=1)
        else:
            segs = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(segs)])

    def resample_waypoints(self, xy, v, l, spacing, loop):
        if loop:
            xy_work = np.vstack([xy, xy[0]])
            v_work = np.concatenate([v, v[:1]])
            l_work = np.concatenate([l, l[:1]])
        else:
            xy_work, v_work, l_work = xy.copy(), v.copy(), l.copy()

        segs = np.linalg.norm(np.diff(xy_work, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(segs)])
        total = float(cum[-1])
        if total < spacing:
            raise ValueError('Waypoint path is too short after preprocessing.')

        if loop:
            sample_s = np.arange(0.0, total, spacing, dtype=np.float64)
        else:
            sample_s = np.arange(0.0, total + 0.5 * spacing, spacing, dtype=np.float64)
            if sample_s[-1] > total:
                sample_s[-1] = total

        x_new = np.interp(sample_s, cum, xy_work[:, 0])
        y_new = np.interp(sample_s, cum, xy_work[:, 1])
        v_new = np.interp(sample_s, cum, v_work)
        l_new = np.interp(sample_s, cum, l_work)

        if not loop and (x_new[-1] != xy[-1, 0] or y_new[-1] != xy[-1, 1]):
            x_new = np.append(x_new, xy[-1, 0])
            y_new = np.append(y_new, xy[-1, 1])
            v_new = np.append(v_new, v[-1])
            l_new = np.append(l_new, l[-1])

        return (
            np.column_stack([x_new, y_new]).astype(np.float64),
            np.asarray(v_new, dtype=np.float64),
            np.asarray(l_new, dtype=np.float64),
        )

    @staticmethod
    def smooth_path(xy: np.ndarray, window_size: int, loop: bool) -> np.ndarray:
        if window_size <= 1 or xy.shape[0] < 3:
            return xy
        if window_size % 2 == 0:
            window_size += 1
        half = window_size // 2
        out = np.zeros_like(xy)
        if loop:
            for i in range(xy.shape[0]):
                idx = [(i + k) % xy.shape[0] for k in range(-half, half + 1)]
                out[i] = np.mean(xy[idx], axis=0)
        else:
            padded = np.pad(xy, ((half, half), (0, 0)), mode='edge')
            for i in range(xy.shape[0]):
                out[i] = np.mean(padded[i:i + window_size], axis=0)
        return out

    # ==================================================================
    # ROS callbacks / control loop
    # ==================================================================
    def odom_callback(self, msg: Odometry):
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        yaw = self.yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_pose = (px, py, yaw)
        self.current_speed = float(msg.twist.twist.linear.x)
        self.last_pose_stamp = msg.header.stamp

    def control_loop(self):
        now = self.get_clock().now()

        if self.current_pose is None or self.last_pose_stamp is None:
            self.get_logger().warn('STOP: no pose received yet', throttle_duration_sec=2.0)
            self.publish_stop(now.to_msg())
            return

        pose_time = rclpy.time.Time.from_msg(self.last_pose_stamp)
        age = (now - pose_time).nanoseconds * 1e-9
        if age > self.cfg.pose_timeout:
            if not self.warned_stale:
                self.get_logger().warn(f'Pose timeout: {age:.3f} s old. Publishing stop.')
                self.warned_stale = True
            self.publish_stop(now.to_msg())
            return
        self.warned_stale = False

        px, py, yaw = self.current_pose
        nearest_idx, nearest_dist = self.find_nearest_index(px, py)
        if nearest_dist > self.cfg.max_reacquire_dist:
            if not self.warned_lost:
                wp = self.waypoints_xy[nearest_idx]
                self.get_logger().warn(
                    f'Off track: nearest wp idx={nearest_idx} '
                    f'({wp[0]:.2f}, {wp[1]:.2f}), dist={nearest_dist:.2f} m. Stop.'
                )
                self.warned_lost = True
            self.publish_stop(now.to_msg())
            return
        self.warned_lost = False

        self.nearest_idx = nearest_idx
        lookahead = float(self.waypoint_lookaheads[nearest_idx])
        goal_idx = self.find_goal_index(nearest_idx, lookahead)
        goal_xy = self.waypoints_xy[goal_idx]
        goal_speed = float(self.waypoint_speeds[goal_idx])
        self.last_goal_idx = goal_idx

        goal_local = self.transform_to_vehicle_frame(goal_xy, px, py, yaw)
        x_local = float(goal_local[0])
        y_local = float(goal_local[1] + self.cfg.target_lateral_offset)

        if x_local < 0.05:
            if self.cfg.loop:
                found_ahead = False
                for step in range(1, min(30, self.num_waypoints)):
                    test_idx = (goal_idx + step) % self.num_waypoints
                    test_local = self.transform_to_vehicle_frame(
                        self.waypoints_xy[test_idx], px, py, yaw
                    )
                    if float(test_local[0]) > 0.05:
                        goal_idx = test_idx
                        goal_xy = self.waypoints_xy[test_idx]
                        x_local = float(test_local[0])
                        y_local = float(test_local[1] + self.cfg.target_lateral_offset)
                        goal_speed = float(self.waypoint_speeds[goal_idx])
                        found_ahead = True
                        break
                if not found_ahead:
                    self.publish_stop(now.to_msg())
                    return
            else:
                self.publish_stop(now.to_msg())
                return

        lookahead_eff_sq = max(x_local * x_local + y_local * y_local, 1e-6)
        curvature = 2.0 * y_local / lookahead_eff_sq
        desired_steer = math.atan(self.cfg.steer_gain * self.cfg.wheelbase * curvature)
        steer_cmd = self.limit_steering(desired_steer, now)

        speed_target = self.compute_target_speed(curvature, goal_speed)
        speed_cmd = self.limit_speed(speed_target, now)

        if (not self.cfg.loop) and self.cfg.stop_on_path_end:
            if self.remaining_distance_to_end(nearest_idx) < lookahead:
                speed_cmd = 0.0

        self.publish_drive(steer_cmd, speed_cmd, now.to_msg())

        if self.cfg.publish_markers:
            self.publish_visualization(goal_xy, now.to_msg())

    # ==================================================================
    # Tracking helpers
    # ==================================================================
    def find_nearest_index(self, px: float, py: float) -> Tuple[int, float]:
        position = np.array([px, py], dtype=np.float64)
        if self.num_waypoints == 0:
            return 0, float('inf')

        if self.last_goal_idx == 0 and self.current_pose is not None and self.last_steer_time is None:
            distances = np.linalg.norm(self.waypoints_xy - position[None, :], axis=1)
            idx = int(np.argmin(distances))
            return idx, float(distances[idx])

        w = min(self.cfg.nearest_search_window, self.num_waypoints - 1)
        if self.cfg.loop:
            candidate_idx = [(self.last_goal_idx + k) % self.num_waypoints for k in range(-w, w + 1)]
        else:
            lo = max(0, self.last_goal_idx - w)
            hi = min(self.num_waypoints - 1, self.last_goal_idx + w)
            candidate_idx = list(range(lo, hi + 1))

        pts = self.waypoints_xy[candidate_idx]
        distances = np.linalg.norm(pts - position[None, :], axis=1)
        best_local = int(np.argmin(distances))
        return int(candidate_idx[best_local]), float(distances[best_local])

    def find_goal_index(self, nearest_idx: int, lookahead: float) -> int:
        if self.cfg.loop:
            target_s = self.arc_lengths[nearest_idx] + lookahead
            total = self.track_length
            if total <= 1e-6:
                return nearest_idx
            target_s = target_s % total
            idx = int(np.searchsorted(self.arc_lengths, target_s, side='left'))
            return 0 if idx >= self.num_waypoints else idx

        target_s = min(self.arc_lengths[nearest_idx] + lookahead, self.arc_lengths[-1])
        idx = int(np.searchsorted(self.arc_lengths, target_s, side='left'))
        return min(idx, self.num_waypoints - 1)

    def remaining_distance_to_end(self, nearest_idx: int) -> float:
        if self.cfg.loop:
            return float('inf')
        return float(max(self.arc_lengths[-1] - self.arc_lengths[nearest_idx], 0.0))

    def compute_target_speed(self, curvature: float, waypoint_speed: float) -> float:
        base = float(waypoint_speed) if self.cfg.use_waypoint_velocity else float(self.cfg.nominal_speed)
        curve_term = self.cfg.curve_slowdown_gain * (abs(curvature) ** self.cfg.curve_speed_power)
        return float(np.clip(base / (1.0 + curve_term), self.cfg.min_speed, self.cfg.max_speed))

    def limit_steering(self, desired: float, now) -> float:
        desired = float(np.clip(desired, -self.cfg.max_steer, self.cfg.max_steer))
        if self.last_steer_time is None:
            self.last_steer_time = now
            self.last_steer_cmd = desired
            return desired
        dt = max((now - self.last_steer_time).nanoseconds * 1e-9, 1e-3)
        max_delta = self.cfg.max_steer_rate * dt
        delta = float(np.clip(desired - self.last_steer_cmd, -max_delta, max_delta))
        self.last_steer_cmd = float(np.clip(
            self.last_steer_cmd + delta, -self.cfg.max_steer, self.cfg.max_steer
        ))
        self.last_steer_time = now
        return self.last_steer_cmd

    def limit_speed(self, target: float, now) -> float:
        target = float(np.clip(target, 0.0, self.cfg.max_speed))
        if self.last_speed_time is None:
            self.last_speed_time = now
            self.last_speed_cmd = target
            return target
        dt = max((now - self.last_speed_time).nanoseconds * 1e-9, 1e-3)
        max_delta = (self.cfg.max_accel if target >= self.last_speed_cmd else self.cfg.max_decel) * dt
        delta = float(np.clip(target - self.last_speed_cmd, -max_delta, max_delta))
        self.last_speed_cmd = float(np.clip(self.last_speed_cmd + delta, 0.0, self.cfg.max_speed))
        self.last_speed_time = now
        return self.last_speed_cmd

    # ==================================================================
    # Geometry helpers
    # ==================================================================
    @staticmethod
    def transform_to_vehicle_frame(point_global: np.ndarray, px: float, py: float, yaw: float):
        dx = float(point_global[0] - px)
        dy = float(point_global[1] - py)
        return np.array([
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
        ], dtype=np.float64)

    @staticmethod
    def yaw_from_quaternion(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
    # ==================================================================
    # Publishers
    # ==================================================================
    def publish_drive(self, steering: float, speed: float, stamp_msg):
        msg = AckermannDriveStamped()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = self.cfg.global_frame
        msg.drive.steering_angle = float(np.clip(steering, -self.cfg.max_steer, self.cfg.max_steer))
        msg.drive.speed = float(np.clip(speed, 0.0, self.cfg.max_speed))
        self.drive_pub.publish(msg)

    def publish_stop(self, stamp_msg):
        msg = AckermannDriveStamped()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = self.cfg.global_frame
        msg.drive.steering_angle = 0.0
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)
        self.last_speed_cmd = 0.0

    def publish_visualization(self, goal_xy: np.ndarray, stamp_msg):
        markers = MarkerArray()

        path_marker = Marker()
        path_marker.header.stamp = stamp_msg
        path_marker.header.frame_id = self.cfg.global_frame
        path_marker.ns = 'pure_pursuit'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05
        path_marker.color.a = 1.0
        path_marker.color.g = 1.0
        path_marker.pose.orientation.w = 1.0
        for wp in self.waypoints_xy:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            path_marker.points.append(p)
        if self.cfg.loop and self.num_waypoints > 1:
            p = Point()
            p.x = float(self.waypoints_xy[0, 0])
            p.y = float(self.waypoints_xy[0, 1])
            path_marker.points.append(p)
        markers.markers.append(path_marker)

        goal_marker = Marker()
        goal_marker.header.stamp = stamp_msg
        goal_marker.header.frame_id = self.cfg.global_frame
        goal_marker.ns = 'pure_pursuit'
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(goal_xy[0])
        goal_marker.pose.position.y = float(goal_xy[1])
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.25
        goal_marker.scale.y = 0.25
        goal_marker.scale.z = 0.25
        goal_marker.color.a = 1.0
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.2
        markers.markers.append(goal_marker)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)

    # Set default waypoints path to the CSV installed with this package
    if not CFG.waypoints_path:
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('pure_pursuit')
            CFG.waypoints_path = os.path.join(share, 'waypoints', 'race3.csv')
        except Exception:
            pass

    node = PurePursuitNode(CFG)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop(node.get_clock().now().to_msg())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

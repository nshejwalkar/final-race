#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import tf2_ros

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid


class ReactiveFollowGap(Node):
    """Reactive obstacle avoidance using the (F1TENTH) Follow-the-Gap method.

    Pipeline (per scan):
      1) Preprocess scan (smooth + clamp)
      2) Find closest point
      3) Apply a safety bubble around the closest point (set to 0)
      4) Find the maximum-length gap (longest consecutive non-zero-ish segment)
      5) Pick a goal point inside the gap (cap far ranges to reduce wiggling)
      6) Publish AckermannDriveStamped

    This version adds:
      - Disparity Extender (inflate obstacle edges by vehicle half-width + margin)
    while keeping your original bubble + max-gap framework.
    """

    def __init__(self):
        super().__init__('reactive_node')

        # ---------------- Topics ----------------
        lidarscan_topic = '/scan'
        drive_topic = '/drive_gap'
        pp_topic = '/drive_pp'

        # ---------------- Tunable parameters ----------------
        # Only plan using the forward field-of-view (front +/- fov_deg/2).
        self.fov_deg = 100.0

        # Preprocess
        self.smoothing_window = 7           # odd number recommended
        self.max_range_clip = 2.0           # clamp far readings (m)

        # Safety bubble (meters). Around the closest obstacle point.
        self.bubble_radius_m = 0.30

        # Gap definition: treat ranges <= gap_min_dist as blocked (0)
        self.gap_min_dist = 0.30

        # Best-point selection: cap far distances to reduce oscillations ("wiggling").
        self.goal_range_cap = 2.0
        self.centering_weight = 0.006       # bias toward the gap center

        # Steering / speed
        self.max_steer = 0.4189             # ~24 deg
        self.max_steer_delta = 0.10         # slew-rate per scan (rad)

        self.speed_straight = 4.0
        self.speed_medium = 1.0
        self.speed_turn = 0.6
        self.min_speed = 0.3

        # Side-collision guard when turning (tweak inspired by lecture "cornering")
        self.side_safe_dist = 0.30          # m

        # Slow down when something is very close in the forward direction
        self.front_window_deg = 20.0
        self.front_slow_dist = 0.5          # m
        self.front_stop_dist = 0.10         # m

        # ---------------- NEW: Disparity Extender parameters ----------------
        # Inflate obstacle edges by vehicle half-width (+ margin)
        self.car_width_m = 0.25             # vehicle width (F1TENTH ~0.30m)
        self.safety_margin_m = 0.05         # extra inflation margin
        self.disparity_thresh = 0.30        # range jump to detect an edge (m)

        # ---------------- Obstacle detection for drive mux ----------------
        self.trigger_dist = 1.5   # switch to gap follow when obstacle this close
        self.clear_dist = 1.5     # switch back to PP only when this far
        self.obstacle_state = False
        self.map_topic = '/map'
        self.wall_radius_cells = 4
        self.wall_occupancy_thresh = 50

        # State
        self._prev_steer = 0.0
        self.last_pp_steer = 0.0
        self.map_grid = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None
        self.map_frame = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- ROS I/O ----------------
        self.scan_sub = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            qos_profile_sensor_data,
        )
        self.pp_sub = self.create_subscription(
            AckermannDriveStamped,
            pp_topic,
            self._on_pp_drive,
            10,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self._on_map,
            1,
        )
        self.obstacle_pub = self.create_publisher(Bool, '/obstacle_ahead', 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.get_logger().info("ReactiveFollowGap node started: Follow-the-Gap is running (with Disparity Extender).")

    def detect_obstacle_blob(
        self,
        ranges,
        angles,
        cone_center: float,
        dist_thresh: float,
        scan_frame: str,
        scan_stamp,
    ) -> bool:
        """Detect a discrete obstacle (not a wall) in the steering-biased front cone."""

        # 1. Take only front cone (biased by steering)
        cone_half = math.radians(15.0)
        front_mask = np.abs(angles - cone_center) < cone_half
        if not np.any(front_mask):
            return False

        front_ranges = ranges[front_mask].copy()
        front_ranges[~np.isfinite(front_ranges)] = 100.0  # treat invalid as far
        front_ranges[front_ranges <= 0.0] = 100.0

        if self._map_ready():
            transform = self._lookup_map_transform(scan_frame, scan_stamp)
            if transform is not None:
                tx, ty, yaw = transform
                front_idx = np.flatnonzero(front_mask)
                for local_i, scan_i in enumerate(front_idx):
                    r = float(front_ranges[local_i])
                    if r >= dist_thresh:
                        continue
                    x_l = r * math.cos(float(angles[scan_i]))
                    y_l = r * math.sin(float(angles[scan_i]))
                    x_w = tx + math.cos(yaw) * x_l - math.sin(yaw) * y_l
                    y_w = ty + math.sin(yaw) * x_l + math.cos(yaw) * y_l
                    if self._is_wall_return(x_w, y_w):
                        front_ranges[local_i] = 100.0
        
        # 2. Find continuous "close" segments (potential obstacles)
        close_mask = front_ranges < dist_thresh
        if not np.any(close_mask):
            return False
        
        # 3. Identify connected blobs in the close mask
        # A blob = run of consecutive True values
        diffs = np.diff(close_mask.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        
        # Handle edges
        if close_mask[0]:
            starts = np.r_[0, starts]
        if close_mask[-1]:
            ends = np.r_[ends, len(close_mask)]
        
        # 4. For each blob, check if it has gaps on EITHER side (= real obstacle)
        for s, e in zip(starts, ends):
            blob_size = e - s
            
            # # Reject blobs that are too small (noise) or too wide (wall)
            if blob_size < 5:
                continue
            if blob_size > 30:  # wider than ~15deg of close returns = probably wall
                continue
            
            # Check if there's free space on at least one side
            # Look at the 5 beams just outside the blob
            left_check = front_ranges[max(0, s-5):s]
            right_check = front_ranges[e:min(len(front_ranges), e+5)]
            
            blob_min = front_ranges[s:e].min()
            
            left_clear = left_check.size > 0 and left_check.mean() > blob_min + 0.8
            right_clear = right_check.size > 0 and right_check.mean() > blob_min + 0.8
            
            if left_clear or right_clear:
                return True  # Real obstacle: close blob with free space beside it
        
        return False

    def _on_map(self, msg: OccupancyGrid) -> None:
        if msg.info.width == 0 or msg.info.height == 0:
            return
        grid = np.array(msg.data, dtype=np.int16)
        if grid.size != msg.info.width * msg.info.height:
            return
        self.map_grid = grid.reshape((msg.info.height, msg.info.width))
        self.map_resolution = float(msg.info.resolution)
        self.map_origin = (float(msg.info.origin.position.x), float(msg.info.origin.position.y))
        self.map_width = int(msg.info.width)
        self.map_height = int(msg.info.height)
        self.map_frame = msg.header.frame_id

    def _map_ready(self) -> bool:
        return (
            self.map_grid is not None
            and self.map_resolution is not None
            and self.map_origin is not None
            and self.map_width is not None
            and self.map_height is not None
            and self.map_frame
        )

    def _lookup_map_transform(self, scan_frame: str, scan_stamp):
        if not self.map_frame or not scan_frame:
            return None
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                scan_frame,
                rclpy.time.Time.from_msg(scan_stamp),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return (float(t.transform.translation.x), float(t.transform.translation.y), float(yaw))

    def _is_wall_return(self, x_w: float, y_w: float) -> bool:
        if not self._map_ready():
            return False
        ox, oy = self.map_origin
        res = self.map_resolution
        cx = int((x_w - ox) / res)
        cy = int((y_w - oy) / res)
        radius = int(self.wall_radius_cells)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                ix = cx + dx
                iy = cy + dy
                if 0 <= ix < self.map_width and 0 <= iy < self.map_height:
                    if self.map_grid[iy, ix] >= self.wall_occupancy_thresh:
                        return True
        return False
    def preprocess_lidar(self, ranges: np.ndarray, range_max: float) -> np.ndarray:
        """Preprocess the LiDAR scan array.

        - Replace inf/NaN
        - Clamp far values
        - Smooth via moving average
        """
        r = np.asarray(ranges, dtype=np.float32)

        # Replace invalids
        r = np.nan_to_num(r, nan=0.0, posinf=range_max, neginf=0.0)

        # Clamp
        r = np.clip(r, 0.0, self.max_range_clip)

        # Smooth (moving average)
        w = int(self.smoothing_window)
        if w >= 3:
            if w % 2 == 0:
                w += 1
            kernel = np.ones(w, dtype=np.float32) / float(w)
            r = np.convolve(r, kernel, mode='same')

        return r

    # ---------------- NEW: Disparity Extender ----------------
    def apply_disparity_extender(self, ranges: np.ndarray, angle_inc: float) -> np.ndarray:
        """
        Disparity Extender:
        When a big range jump is detected, extend the *closer* distance into the
        *farther* side by enough beams to account for vehicle half-width (+ margin).

        This reduces "fake wide gaps" near obstacle edges that can trap the car.
        """
        r = ranges.copy()
        n = len(r)
        if n < 3:
            return r

        half_width = 0.5 * self.car_width_m + self.safety_margin_m
        if half_width <= 0.0:
            return r

        diffs = np.abs(np.diff(r))
        edge_idxs = np.where(diffs > self.disparity_thresh)[0]  # edge between i and i+1

        for i in edge_idxs:
            d1 = float(r[i])
            d2 = float(r[i + 1])

            # Skip invalid / blocked rays
            if d1 <= 1e-6 or d2 <= 1e-6:
                continue

            # Determine which side is closer, extend into farther side
            if d1 < d2:
                close_d = d1
                start = i + 1
                step = +1
            else:
                close_d = d2
                start = i
                step = -1

            close_d = max(close_d, 1e-3)

            # how many beams to extend so lateral clearance >= half_width
            ratio = half_width / close_d
            if ratio >= 1.0:
                k = n
            else:
                theta = float(np.arcsin(ratio))  # angular half-span
                k = int(np.ceil(theta / max(angle_inc, 1e-6)))

            for t in range(k + 1):
                j = start + step * t
                if j < 0 or j >= n:
                    break
                r[j] = min(r[j], close_d)

        return r

    def find_max_gap(self, free_space_ranges: np.ndarray):
        """Return (start_i, end_i) inclusive indices of the max gap."""
        free = free_space_ranges > self.gap_min_dist
        if not np.any(free):
            return None

        idx = np.flatnonzero(free)
        # Split into contiguous segments
        cut = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[cut + 1]]
        ends = np.r_[idx[cut], idx[-1]]

        lengths = ends - starts
        best_k = int(np.argmax(lengths))
        return int(starts[best_k]), int(ends[best_k])

    def find_best_point(self, start_i: int, end_i: int, ranges: np.ndarray) -> int:
        """Pick the best point inside [start_i, end_i].

        We cap far distances and add a small bias toward the gap center.
        This is a simple version of the "reduce wiggling" tweak.
        """
        seg = ranges[start_i:end_i + 1]
        if seg.size == 0:
            return int((start_i + end_i) // 2)

        seg_cap = np.minimum(seg, self.goal_range_cap)
        seg_idx = np.arange(start_i, end_i + 1, dtype=np.float32)
        center = 0.5 * (start_i + end_i)

        # Score: deeper is better, but discourage hugging gap edges
        score = seg_cap - self.centering_weight * np.abs(seg_idx - center)
        best_rel = int(np.argmax(score))
        return int(start_i + best_rel)

    def _slew_limit(self, steer: float) -> float:
        delta = steer - self._prev_steer
        delta = float(np.clip(delta, -self.max_steer_delta, self.max_steer_delta))
        steer_limited = self._prev_steer + delta
        self._prev_steer = steer_limited
        return steer_limited

    def _on_pp_drive(self, msg: AckermannDriveStamped) -> None:
        self.last_pp_steer = float(msg.drive.steering_angle)

    def lidar_callback(self, data: LaserScan):
        """Process each LiDAR scan and publish a drive command."""

        ranges = np.asarray(data.ranges, dtype=np.float32)
        n = len(ranges)
        angle_min = float(data.angle_min)
        angle_inc = float(data.angle_increment)
        angles = angle_min + np.arange(n, dtype=np.float32) * angle_inc
        cone_center = float(self.last_pp_steer)

        # --- obstacle detection ---
        detected = self.detect_obstacle_blob(
            ranges,
            angles,
            cone_center,
            self.trigger_dist,
            data.header.frame_id,
            data.header.stamp,
        )
        cleared = self.detect_obstacle_blob(
            ranges,
            angles,
            cone_center,
            self.clear_dist,
            data.header.frame_id,
            data.header.stamp,
        )

        if detected:
            self.obstacle_state = True
        elif not cleared:
            self.obstacle_state = False

        bool_msg = Bool()
        bool_msg.data = self.obstacle_state
        self.obstacle_pub.publish(bool_msg)
        if n == 0:
            return

        range_max = float(data.range_max) if np.isfinite(data.range_max) else 10.0
        raw = np.asarray(data.ranges, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=range_max, neginf=0.0)

        proc = self.preprocess_lidar(raw, range_max)

        angles = data.angle_min + np.arange(n, dtype=np.float32) * data.angle_increment

        # Limit planning to front FOV
        fov = np.deg2rad(self.fov_deg * 0.5)
        front_mask = (angles >= -fov) & (angles <= fov)
        proc[~front_mask] = 0.0

        # -------- NEW: apply disparity extender (still within front FOV) --------
        proc_ext = self.apply_disparity_extender(proc, float(data.angle_increment))

        # Find closest point (within front FOV)  (use proc for stability)
        valid = proc > 0.0
        if not np.any(valid):
            return

        closest_i = int(np.argmin(np.where(valid, proc, np.inf)))
        closest_dist = float(proc[closest_i])

        # Safety bubble around closest point (convert meters to indices)
        # half-angle ~= atan(rb / d)
        d = max(closest_dist, 1e-3)
        half_angle = float(np.arctan2(self.bubble_radius_m, d))
        bubble = int(max(1, half_angle / max(float(data.angle_increment), 1e-6)))

        left = max(0, closest_i - bubble)
        right = min(n - 1, closest_i + bubble)
        proc_ext[left:right + 1] = 0.0

        # Max gap (use proc_ext)
        gap = self.find_max_gap(proc_ext)
        if gap is None:
            best_i = int(np.argmin(np.abs(angles)))  # closest to 0 rad
        else:
            start_i, end_i = gap
            best_i = self.find_best_point(start_i, end_i, proc_ext)

        # Steering toward the chosen lidar ray
        target_angle = float(angles[best_i])
        steer = float(np.clip(target_angle, -self.max_steer, self.max_steer))

        # Side-collision guard (if turning, ensure side/back on that side isn't too close)
        left_side = raw[angles > (np.pi / 2.0)]
        right_side = raw[angles < (-np.pi / 2.0)]
        min_left = float(np.min(left_side)) if left_side.size else np.inf
        min_right = float(np.min(right_side)) if right_side.size else np.inf
        if steer > 0.0 and min_left < self.side_safe_dist:
            steer = 0.0
        if steer < 0.0 and min_right < self.side_safe_dist:
            steer = 0.0

        # Slew-rate limit steering (helps stability)
        steer = self._slew_limit(steer)

        # Speed selection based on steering magnitude
        a = abs(steer)
        if a < 0.10:
            speed = self.speed_straight
        elif a < 0.25:
            speed = self.speed_medium
        else:
            speed = self.speed_turn

        # Additional slowdown based on forward clearance
        fw = np.deg2rad(self.front_window_deg * 0.5)
        fw_mask = (angles >= -fw) & (angles <= fw)
        front_min = float(np.min(np.where(fw_mask, raw, np.inf)))
        if np.isfinite(front_min):
            if front_min < self.front_stop_dist:
                speed = self.min_speed
            elif front_min < self.front_slow_dist:
                # Linear ramp between slow_dist and stop_dist
                t = (front_min - self.front_stop_dist) / max(self.front_slow_dist - self.front_stop_dist, 1e-3)
                speed = min(speed, self.min_speed + t * (self.speed_turn - self.min_speed))

        speed = float(max(self.min_speed, speed))

        # Publish
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
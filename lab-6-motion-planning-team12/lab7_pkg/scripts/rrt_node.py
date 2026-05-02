#!/usr/bin/env python3
"""
RRT* hybrid racer (waypoints with per-point velocity)
=====================================================
Waypoint CSV 格式: x, y, v  (第三列没有则回退到 default_waypoint_speed)

高层思路:
  - 默认走纯跟踪,速度直接来自 waypoint 第三列(乘以全局 velocity_scale)。
  - 检测到障碍才跑 RRT*;此时速度 = min(waypoint速度, obstacle_speed)。
"""

import csv
import math
import random
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class RRTStarNode(Node):
    def __init__(self):
        super().__init__('rrt_star_node')

        # ====================================================================
        # 配置
        # ====================================================================
        self.sim_mode = True
        self.plan_frequency = 20.0

        self.waypoints_file = ""

        # ----- RRT* -----
        self.max_expansion_dist = 0.45
        self.max_iterations = 300
        self.early_term_iters = 60
        self.goal_threshold = 0.30
        self.goal_bias_prob = 0.35
        self.search_radius = 0.9
        self.max_shortcut_dist = 1.6

        # ----- 目标选择 -----
        self.lookahead_dist = 2.0
        self.max_lookahead_dist = 4.5

        # ----- 采样窗口 -----
        self.sample_x_min = -0.3
        self.sample_x_max = 4.5
        self.sample_y_max = 2.0
        self.forward_sample_ratio = 0.85

        # ----- Occupancy grid -----
        self.grid_resolution = 0.10
        self.grid_width = 200
        self.grid_height = 200
        self.inflation_radius = 2
        self.car_chassis_radius = 0.25

        # ----- 控制 -----
        self.wheelbase = 0.33
        self.pursuit_lookahead_clear = 1.20
        self.pursuit_lookahead_avoid = 0.70
        self.max_steer = 0.40

        # ----- 速度 -----
        # 速度主要来自 waypoint 第三列。下面这些是 cap / fallback / 缩放。
        self.velocity_scale = 1.0            # 全局速度缩放(初次上车建议 0.7)
        self.default_waypoint_speed = 2.0    # waypoint 缺第三列时的默认速度
        self.max_speed = 10.0                 # 速度硬上限(防 CSV 异常值)
        self.min_speed = 0.6
        self.obstacle_speed = 3.0            # 避障模式速度上限
        self.safe_fallback_speed = 0.35
        # waypoint 已编码曲率,所以这里只做小幅修正
        self.curvature_speed_gain = 0.3

        # ----- 模式滞回 -----
        self.path_clear_check_dist = 3.5
        self.replan_hysteresis_cycles = 4

        # ----- 帧 / 话题 -----
        self.base_frame = 'ego_racecar/base_link' if self.sim_mode else 'base_link'
        self.map_frame = 'map'
        self.odom_topic = '/ego_racecar/odom' if self.sim_mode else '/pf/pose/odom'
        self.scan_topic = '/scan'
        self.drive_topic = '/drive'

        # ====================================================================
        # 状态
        # ====================================================================
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_yaw = 0.0
        self.cos_yaw = 1.0
        self.sin_yaw = 0.0
        self.goal_local_x = 0.0
        self.goal_local_y = 0.0
        self.target_waypoint_speed = self.default_waypoint_speed   # 当前目标点速度
        self.pose_ready = False
        self.scan_ready = False
        self.pose_tick = 0

        self.in_avoidance_mode = False
        self.avoidance_ttl = 0
        self.last_closest_wp_idx = 0

        self.waypoints = np.empty((0, 3))    # (N, 3): x, y, v
        self.occupancy_grid = np.zeros(self.grid_width * self.grid_height, dtype=np.uint8)

        self._init_tree_arrays()
        self._build_inflation_kernel()
        self.load_waypoints()

        # ====================================================================
        # ROS 接口
        # ====================================================================
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)
        self.tree_pub = self.create_publisher(Marker, '/rrt_tree', 1)
        self.path_pub = self.create_publisher(Marker, '/rrt_path', 1)
        self.waypoints_pub = self.create_publisher(Marker, '/rrt_waypoints', 1)
        self.goal_pub = self.create_publisher(Marker, '/rrt_goal', 1)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/rrt_occupancy_grid', 1)

        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)
        self.plan_timer = self.create_timer(1.0 / self.plan_frequency, self.plan_callback)

        self.use_rrt_star = True

        self.get_logger().info(
            f'RRT* hybrid racer started. mode={"sim" if self.sim_mode else "real"}, '
            f'odom={self.odom_topic}, velocity_scale={self.velocity_scale}'
        )

    # ========================================================================
    # 初始化辅助
    # ========================================================================
    def _init_tree_arrays(self):
        N = self.max_iterations + 1
        self.tx = np.zeros(N, dtype=np.float64)
        self.ty = np.zeros(N, dtype=np.float64)
        self.tparent = np.full(N, -1, dtype=np.int32)
        self.tcost = np.zeros(N, dtype=np.float64)
        self.tree_size = 0

    def _build_inflation_kernel(self):
        r = self.inflation_radius
        dx_list, dy_list = [], []
        r2 = r * r
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r2:
                    dx_list.append(dx)
                    dy_list.append(dy)
        self._inflate_dx = np.array(dx_list, dtype=np.int32)
        self._inflate_dy = np.array(dy_list, dtype=np.int32)

    # ========================================================================
    # Waypoints (x, y, v)
    # ========================================================================
    def resolve_waypoints_file(self):
        if self.waypoints_file:
            candidate = Path(self.waypoints_file).expanduser()
            if candidate.is_file():
                return candidate
            self.get_logger().warn(f'Configured waypoint file does not exist: {candidate}')

        search_dirs = []
        try:
            script_dir = Path(__file__).resolve().parent
            search_dirs.extend([
                script_dir,
                script_dir.parent,
                script_dir.parent / 'waypoints',
                script_dir.parent / 'logs',
            ])
        except Exception:
            pass

        home = Path.home()
        search_dirs.extend([
            home / 'roboracer_ws' / 'src' / 'lab-6-motion-planning-team12' / 'waypoints',
            home / 'rcws' / 'logs',
            Path.cwd(),
        ])

        csv_files = []
        for directory in search_dirs:
            if directory.is_dir():
                csv_files.extend(directory.glob('*.csv'))

        if not csv_files:
            return None
        return max(csv_files, key=lambda p: p.stat().st_mtime)

    def load_waypoints(self):
        path = self.resolve_waypoints_file()
        if path is None:
            self.get_logger().warn('No waypoints CSV found.')
            return

        loaded, skipped, with_speed = [], 0, 0
        try:
            with path.open('r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if not row:
                        continue
                    first = str(row[0]).strip()
                    if not first or first.startswith('#'):
                        continue
                    if len(row) < 2:
                        skipped += 1
                        continue
                    try:
                        x = float(row[0]); y = float(row[1])
                    except ValueError:
                        skipped += 1
                        continue

                    # 第三列速度,取不到就用默认值
                    v = self.default_waypoint_speed
                    if len(row) >= 3:
                        try:
                            v = float(row[2])
                            with_speed += 1
                        except ValueError:
                            pass
                    loaded.append([x, y, v])

            self.waypoints = np.array(loaded, dtype=np.float64) if loaded else np.empty((0, 3))
            if self.waypoints.shape[0]:
                vmin = float(self.waypoints[:, 2].min())
                vmax = float(self.waypoints[:, 2].max())
                self.get_logger().info(
                    f'Loaded {len(self.waypoints)} waypoints from {path} '
                    f'(with_speed={with_speed}, skipped={skipped}, v_range=[{vmin:.2f}, {vmax:.2f}]).'
                )
            else:
                self.get_logger().warn(f'No usable waypoints in {path}.')
        except Exception as e:
            self.get_logger().warn(f'Failed to load waypoints: {e}')

    # ========================================================================
    # ROS 回调
    # ========================================================================
    def pose_callback(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.cos_yaw = math.cos(self.car_yaw)
        self.sin_yaw = math.sin(self.car_yaw)
        self.pose_ready = True

        if self.pose_tick % 20 == 0:
            self.visualize_waypoints()
        self.pose_tick += 1

    def scan_callback(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        if ranges.size == 0:
            return
        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment

        valid = (
            np.isfinite(ranges)
            & (ranges >= max(msg.range_min, self.car_chassis_radius))
            & (ranges <= msg.range_max)
        )
        self.occupancy_grid.fill(0)

        if not np.any(valid):
            self.scan_ready = True
            self.publish_occupancy_grid()
            return

        r = ranges[valid]; a = angles[valid]
        xs = r * np.cos(a); ys = r * np.sin(a)

        half_w = self.grid_width // 2
        half_h = self.grid_height // 2
        gx = np.rint(xs / self.grid_resolution).astype(np.int32) + half_w
        gy = np.rint(ys / self.grid_resolution).astype(np.int32) + half_h
        in_b = (gx >= 0) & (gx < self.grid_width) & (gy >= 0) & (gy < self.grid_height)
        gx = gx[in_b]; gy = gy[in_b]

        if gx.size > 0:
            all_gx = (gx[:, None] + self._inflate_dx[None, :]).ravel()
            all_gy = (gy[:, None] + self._inflate_dy[None, :]).ravel()
            in_b2 = (all_gx >= 0) & (all_gx < self.grid_width) & (all_gy >= 0) & (all_gy < self.grid_height)
            if np.any(in_b2):
                idx = all_gy[in_b2] * self.grid_width + all_gx[in_b2]
                self.occupancy_grid[idx] = 1

        self.scan_ready = True
        self.publish_occupancy_grid()

    # ========================================================================
    # 主规划回调
    # ========================================================================
    def plan_callback(self):
        if not (self.pose_ready and self.scan_ready) or self.waypoints.shape[0] == 0:
            return

        ok, gx, gy, blocked = self.find_goal()
        if not ok:
            return
        self.goal_local_x = gx
        self.goal_local_y = gy
        self.visualize_goal(gx, gy)

        need_avoidance = blocked or not self.raceline_clear()
        if need_avoidance:
            if not self.in_avoidance_mode:
                reason = "blocked" if blocked else "raceline not clear"
                self.get_logger().info(f'[MODE] Switching to RRT* avoidance ({reason})')
            self.in_avoidance_mode = True
            self.avoidance_ttl = self.replan_hysteresis_cycles
        elif self.avoidance_ttl > 0:
            self.avoidance_ttl -= 1
            self.in_avoidance_mode = self.avoidance_ttl > 0
        else:
            if self.in_avoidance_mode:
                self.get_logger().info('[MODE] Switching back to pure pursuit')
            self.in_avoidance_mode = False

        if self.in_avoidance_mode:
            self.run_rrt_star_and_drive(gx, gy)
        else:
            self.run_pure_pursuit_on_waypoints()
            self.visualize_path([])
            self.clear_tree_marker()

    # ========================================================================
    # 速度封装
    # ========================================================================
    def clamp_speed(self, v, cap=None):
        v = v * self.velocity_scale
        if cap is not None:
            v = min(v, cap)
        return max(self.min_speed, min(self.max_speed, v))

    # ========================================================================
    # 快路径:waypoint 纯跟踪(速度来自 waypoint 第三列)
    # ========================================================================
    def raceline_clear(self):
        if self.waypoints.shape[0] == 0:
            return True
        N = self.waypoints.shape[0]
        idx = self.last_closest_wp_idx
        accumulated = 0.0
        prev_lx = prev_ly = None
        max_steps = min(N, 80)

        for _ in range(max_steps):
            wp = self.waypoints[idx]
            dx = wp[0] - self.car_x
            dy = wp[1] - self.car_y
            lx = self.cos_yaw * dx + self.sin_yaw * dy
            ly = -self.sin_yaw * dx + self.cos_yaw * dy
            if lx > -0.3:
                if prev_lx is None:
                    prev_lx, prev_ly = max(0.0, lx), ly
                else:
                    if self.line_collision(prev_lx, prev_ly, lx, ly):
                        return False
                    accumulated += math.hypot(lx - prev_lx, ly - prev_ly)
                    prev_lx, prev_ly = lx, ly
                    if accumulated >= self.path_clear_check_dist:
                        return True
            idx = (idx + 1) % N
        return True

    def run_pure_pursuit_on_waypoints(self):
        N = self.waypoints.shape[0]
        if N == 0:
            self.publish_drive(0.0, 0.0)
            return

        # 增量找 closest
        win = min(N, 80)
        start = (self.last_closest_wp_idx - win // 4) % N
        d2_min = float('inf')
        closest = self.last_closest_wp_idx
        for k in range(win):
            i = (start + k) % N
            dx = self.waypoints[i, 0] - self.car_x
            dy = self.waypoints[i, 1] - self.car_y
            d2 = dx * dx + dy * dy
            if d2 < d2_min:
                d2_min = d2
                closest = i
        self.last_closest_wp_idx = closest

        # 找 lookahead 处的目标 waypoint
        target_idx = closest
        target_lx = self.pursuit_lookahead_clear
        target_ly = 0.0
        for k in range(N):
            i = (closest + k) % N
            dx = self.waypoints[i, 0] - self.car_x
            dy = self.waypoints[i, 1] - self.car_y
            lx = self.cos_yaw * dx + self.sin_yaw * dy
            ly = -self.sin_yaw * dx + self.cos_yaw * dy
            if lx > 0.0 and math.hypot(dx, dy) >= self.pursuit_lookahead_clear:
                target_idx = i
                target_lx, target_ly = lx, ly
                break

        steer = self.pure_pursuit_steer(target_lx, target_ly)

        # ★ 速度直接来自该 waypoint
        wp_speed = float(self.waypoints[target_idx, 2])
        # waypoint 已编码曲率,只做小幅转向修正
        steer_factor = 1.0 - self.curvature_speed_gain * abs(steer) / self.max_steer
        speed = self.clamp_speed(wp_speed * steer_factor)
        self.target_waypoint_speed = wp_speed

        self.publish_drive(speed, steer)

    def pure_pursuit_steer(self, lx, ly):
        L2 = lx * lx + ly * ly
        if L2 < 1e-6:
            return 0.0
        steer = math.atan2(2.0 * self.wheelbase * ly, L2)
        return max(-self.max_steer, min(self.max_steer, steer))

    # ========================================================================
    # 慢路径:RRT*
    # ========================================================================
    def run_rrt_star_and_drive(self, gx, gy):
        self.tree_size = 0
        self.add_node(0.0, 0.0, parent=-1, cost=0.0)

        best_goal_idx = -1
        best_goal_cost = float('inf')
        iters_since_first = 0

        for _ in range(self.max_iterations):
            sx, sy = self.sample(gx, gy)
            nearest_idx = self.nearest(sx, sy)
            nx, ny = self.steer_to(nearest_idx, sx, sy)

            if self.line_collision(self.tx[nearest_idx], self.ty[nearest_idx], nx, ny):
                if best_goal_idx >= 0:
                    iters_since_first += 1
                    if iters_since_first >= self.early_term_iters:
                        break
                continue

            new_idx = self.rrt_star_insert(nx, ny, nearest_idx)

            if math.hypot(nx - gx, ny - gy) < self.goal_threshold:
                if self.tcost[new_idx] < best_goal_cost:
                    best_goal_cost = self.tcost[new_idx]
                    best_goal_idx = new_idx

            if best_goal_idx >= 0:
                iters_since_first += 1
                if iters_since_first >= self.early_term_iters:
                    break

            if self.tree_size >= self.tx.size - 1:
                break

        self.visualize_tree()

        if best_goal_idx < 0:
            self.visualize_path([])
            self.safe_fallback_command(gx, gy)
            return

        path = self.extract_path(best_goal_idx)
        path = self.shortcut_path(path)
        self.visualize_path(path)
        self.execute_avoidance_path(path)

    def add_node(self, x, y, parent, cost):
        i = self.tree_size
        self.tx[i] = x; self.ty[i] = y
        self.tparent[i] = parent
        self.tcost[i] = cost
        self.tree_size += 1
        return i

    def sample(self, gx, gy):
        if random.random() < self.goal_bias_prob:
            return gx, gy
        x_max = max(self.sample_x_max, gx + 1.0)
        y_max = max(self.sample_y_max, abs(gy) + 0.5)
        if random.random() < self.forward_sample_ratio:
            x = random.uniform(0.0, x_max)
            y = random.uniform(-self.sample_y_max, self.sample_y_max)
        else:
            x = random.uniform(self.sample_x_min, x_max)
            y = random.uniform(-y_max, y_max)
        return x, y

    def nearest(self, x, y):
        n = self.tree_size
        dx = self.tx[:n] - x
        dy = self.ty[:n] - y
        return int(np.argmin(dx * dx + dy * dy))

    def near(self, x, y):
        n = self.tree_size
        dx = self.tx[:n] - x
        dy = self.ty[:n] - y
        return np.where(dx * dx + dy * dy <= self.search_radius * self.search_radius)[0]

    def steer_to(self, from_idx, sx, sy):
        x0, y0 = self.tx[from_idx], self.ty[from_idx]
        dx = sx - x0; dy = sy - y0
        d = math.hypot(dx, dy)
        if d <= self.max_expansion_dist:
            return float(sx), float(sy)
        s = self.max_expansion_dist / d
        return float(x0 + dx * s), float(y0 + dy * s)

    def rrt_star_insert(self, nx, ny, nearest_idx):
        neighbors = self.near(nx, ny)
        best_parent = nearest_idx
        best_cost = self.tcost[nearest_idx] + math.hypot(
            self.tx[nearest_idx] - nx, self.ty[nearest_idx] - ny
        )
        for idx in neighbors:
            if idx == nearest_idx:
                continue
            cand = self.tcost[idx] + math.hypot(self.tx[idx] - nx, self.ty[idx] - ny)
            if cand + 1e-9 < best_cost and not self.line_collision(self.tx[idx], self.ty[idx], nx, ny):
                best_cost = cand
                best_parent = int(idx)

        new_idx = self.add_node(nx, ny, parent=int(best_parent), cost=float(best_cost))

        for idx in neighbors:
            if idx == best_parent or idx == 0:
                continue
            cand = best_cost + math.hypot(self.tx[idx] - nx, self.ty[idx] - ny)
            if cand + 1e-9 < self.tcost[idx] and not self.line_collision(nx, ny, self.tx[idx], self.ty[idx]):
                delta = self.tcost[idx] - cand
                self.tparent[idx] = new_idx
                self.tcost[idx] = cand
                self.propagate_cost_decrease(int(idx), delta)
        return new_idx

    def propagate_cost_decrease(self, idx, delta):
        n = self.tree_size
        stack = [idx]
        parents = self.tparent[:n]
        while stack:
            cur = stack.pop()
            children = np.where(parents == cur)[0]
            if children.size:
                self.tcost[children] -= delta
                stack.extend(int(c) for c in children)

    # ========================================================================
    # 碰撞检测
    # ========================================================================
    def line_collision(self, x0, y0, x1, y1):
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-6:
            valid, gx, gy = self.world_to_grid(x0, y0)
            if not valid:
                return True
            return bool(self.occupancy_grid[gy * self.grid_width + gx])

        n = max(2, int(math.ceil(d / (self.grid_resolution * 0.7))) + 1)
        ts = np.linspace(0.0, 1.0, n)
        xs = x0 + ts * (x1 - x0)
        ys = y0 + ts * (y1 - y0)
        gx = np.rint(xs / self.grid_resolution).astype(np.int32) + self.grid_width // 2
        gy = np.rint(ys / self.grid_resolution).astype(np.int32) + self.grid_height // 2
        if np.any((gx < 0) | (gx >= self.grid_width) | (gy < 0) | (gy >= self.grid_height)):
            return True
        return bool(np.any(self.occupancy_grid[gy * self.grid_width + gx]))

    # ========================================================================
    # 路径提取 / shortcut
    # ========================================================================
    def extract_path(self, leaf_idx):
        path = []
        idx = leaf_idx
        while idx >= 0:
            path.append((float(self.tx[idx]), float(self.ty[idx])))
            if self.tparent[idx] < 0:
                break
            idx = int(self.tparent[idx])
        return path[::-1]

    def shortcut_path(self, path):
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        N = len(path)
        while i < N - 1:
            best_j = i + 1
            for j in range(N - 1, i + 1, -1):
                if math.hypot(path[i][0] - path[j][0], path[i][1] - path[j][1]) > self.max_shortcut_dist:
                    continue
                if not self.line_collision(path[i][0], path[i][1], path[j][0], path[j][1]):
                    best_j = j
                    break
            out.append(path[best_j])
            i = best_j
        return out

    # ========================================================================
    # 目标选择(顺便记录目标 waypoint 的速度)
    # ========================================================================
    def find_goal(self):
        if self.waypoints.shape[0] == 0:
            return False, 0.0, 0.0, False

        N = self.waypoints.shape[0]

        win = min(N, 80)
        start = (self.last_closest_wp_idx - win // 4) % N
        d2_min = float('inf')
        closest = self.last_closest_wp_idx
        for k in range(win):
            i = (start + k) % N
            dxm = self.waypoints[i, 0] - self.car_x
            dym = self.waypoints[i, 1] - self.car_y
            d2 = dxm * dxm + dym * dym
            if d2 < d2_min:
                d2_min = d2
                closest = i
        self.last_closest_wp_idx = closest

        def to_local(i):
            dxm = self.waypoints[i, 0] - self.car_x
            dym = self.waypoints[i, 1] - self.car_y
            return self.cos_yaw * dxm + self.sin_yaw * dym, -self.sin_yaw * dxm + self.cos_yaw * dym

        def scan_forward(la):
            for k in range(N):
                i = (closest + k) % N
                dxm = self.waypoints[i, 0] - self.car_x
                dym = self.waypoints[i, 1] - self.car_y
                if math.hypot(dxm, dym) >= la:
                    lx, ly = to_local(i)
                    if lx > -0.5:
                        return True, lx, ly, i
            return False, 0.0, 0.0, -1

        ok, lx, ly, idx = scan_forward(self.lookahead_dist)
        if not ok:
            lx, ly = to_local(closest)
            self.target_waypoint_speed = float(self.waypoints[closest, 2])
            return True, lx, ly, False

        valid, gxg, gyg = self.world_to_grid(lx, ly)
        blocked = bool(valid and self.occupancy_grid[gyg * self.grid_width + gxg])
        if not blocked:
            self.target_waypoint_speed = float(self.waypoints[idx, 2])
            return True, lx, ly, False

        # 目标被挡:沿 waypoint 继续往前找空闲点
        for la in (self.lookahead_dist + 0.5, self.lookahead_dist + 1.5, self.max_lookahead_dist):
            ok2, lx2, ly2, idx2 = scan_forward(la)
            if not ok2:
                continue
            v2, gx2, gy2 = self.world_to_grid(lx2, ly2)
            if v2 and not self.occupancy_grid[gy2 * self.grid_width + gx2]:
                self.target_waypoint_speed = float(self.waypoints[idx2, 2])
                return True, lx2, ly2, True

        self.target_waypoint_speed = float(self.waypoints[idx, 2])
        return True, lx, ly, True

    # ========================================================================
    # 路径执行 / fallback
    # ========================================================================
    def execute_avoidance_path(self, path):
        if len(path) < 2:
            self.publish_drive(0.0, 0.0)
            return

        traveled = 0.0
        target_x, target_y = path[-1]
        for i in range(1, len(path)):
            seg = math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            if traveled + seg >= self.pursuit_lookahead_avoid:
                t = (self.pursuit_lookahead_avoid - traveled) / max(seg, 1e-6)
                target_x = path[i - 1][0] + t * (path[i][0] - path[i - 1][0])
                target_y = path[i - 1][1] + t * (path[i][1] - path[i - 1][1])
                break
            traveled += seg

        steer = self.pure_pursuit_steer(target_x, target_y)

        # ★ 速度 = min(目标 waypoint 速度, obstacle_speed),再按转向衰减
        base = min(self.target_waypoint_speed, self.obstacle_speed)
        steer_factor = 1.0 - 0.5 * abs(steer) / max(self.max_steer, 1e-6)
        speed = self.clamp_speed(base * steer_factor, cap=self.obstacle_speed)
        self.publish_drive(speed, steer)

    def safe_fallback_command(self, gx, gy):
        if abs(gy) < 1.0 and not self.line_collision(0.0, 0.0, gx, gy):
            steer = self.pure_pursuit_steer(gx, gy)
            self.publish_drive(self.safe_fallback_speed, steer)
        else:
            self.publish_drive(0.0, 0.0)

    def publish_drive(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)

    # ========================================================================
    # 栅格辅助
    # ========================================================================
    def world_to_grid(self, wx, wy):
        gx = int(round(wx / self.grid_resolution)) + self.grid_width // 2
        gy = int(round(wy / self.grid_resolution)) + self.grid_height // 2
        return (0 <= gx < self.grid_width and 0 <= gy < self.grid_height), gx, gy

    def publish_occupancy_grid(self):
        msg = OccupancyGrid()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(self.grid_resolution)
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = -(self.grid_width / 2.0) * self.grid_resolution
        msg.info.origin.position.y = -(self.grid_height / 2.0) * self.grid_resolution
        msg.info.origin.orientation.w = 1.0
        msg.data = (self.occupancy_grid.astype(np.int8) * 100).tolist()
        self.grid_pub.publish(msg)

    # ========================================================================
    # 可视化
    # ========================================================================
    def visualize_tree(self):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rrt_tree'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.g = 0.8
        marker.color.a = 0.6
        n = self.tree_size
        for i in range(1, n):
            p = int(self.tparent[i])
            if p < 0:
                continue
            marker.points.append(Point(x=float(self.tx[p]), y=float(self.ty[p]), z=0.0))
            marker.points.append(Point(x=float(self.tx[i]), y=float(self.ty[i]), z=0.0))
        self.tree_pub.publish(marker)

    def clear_tree_marker(self):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rrt_tree'
        marker.id = 0
        marker.action = Marker.DELETE
        self.tree_pub.publish(marker)

    def visualize_path(self, path):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rrt_path'
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.06
        marker.color.r = 1.0
        marker.color.a = 1.0
        for p in path:
            marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
        self.path_pub.publish(marker)

    def visualize_waypoints(self):
        if self.waypoints.shape[0] == 0:
            return
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rrt_waypoints'
        marker.id = 2
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.a = 1.0
        for wp in self.waypoints:
            marker.points.append(Point(x=float(wp[0]), y=float(wp[1]), z=0.0))
        self.waypoints_pub.publish(marker)

    def visualize_goal(self, gx, gy):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rrt_goal'
        marker.id = 3
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(gx)
        marker.pose.position.y = float(gy)
        marker.pose.position.z = 0.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 0.4 if self.in_avoidance_mode else 1.0
        marker.color.a = 1.0
        self.goal_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = RRTStarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Monte Carlo Localization (particle filter) for F1Tenth.

Algorithm from L07/L08 slides:
  1. Predict: propagate particles via odometry motion model + noise
  2. Correct: weight particles by LiDAR scan correlation against map
  3. Resample: survival-of-the-fittest (KLD-adaptive particle count)
  4. Publish: best-estimate pose as TF map→odom correction + PoseArray
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import (
    PoseWithCovarianceStamped, PoseArray, Pose,
    TransformStamped, Quaternion,
)
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, Buffer, TransformListener


# ── helpers ────────────────────────────────────────────────────────────────────

def _quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def _yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


# ── node ──────────────────────────────────────────────────────────────────────

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter')

        # ---- parameters ----
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('angle_step', 18)          # subsample every N beams
        self.declare_parameter('squash_factor', 2.2)      # lidar likelihood squash
        self.declare_parameter('max_range', 10.0)         # metres
        self.declare_parameter('motion_noise_x', 0.05)
        self.declare_parameter('motion_noise_y', 0.05)
        self.declare_parameter('motion_noise_theta', 0.05)
        self.declare_parameter('min_particles', 250)
        self.declare_parameter('max_particles', 5000)
        self.declare_parameter('kld_epsilon', 0.05)
        self.declare_parameter('kld_delta', 0.01)
        self.declare_parameter('update_min_d', 0.1)       # m  – minimum travel before update
        self.declare_parameter('update_min_a', 0.1)       # rad
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_a', 0.0)
        self.declare_parameter('initial_cov_xx', 0.5)
        self.declare_parameter('initial_cov_yy', 0.5)
        self.declare_parameter('initial_cov_aa', 0.2)

        p = self._params()

        # ---- state ----
        self._map: OccupancyGrid | None = None
        self._map_data: np.ndarray | None = None   # 2-D occupancy array, 0-100 or -1
        self._particles: np.ndarray | None = None  # (N,3) x,y,theta
        self._weights: np.ndarray | None = None    # (N,)

        # last odom pose for delta computation
        self._last_odom: np.ndarray | None = None  # [x, y, theta]
        self._odom_accum_d = 0.0
        self._odom_accum_a = 0.0
        self._last_linear_vel = 0.0   # pass-through to Odometry output
        self._last_angular_vel = 0.0

        # ---- TF ----
        self._tf_broadcaster = TransformBroadcaster(self)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- subscriptions ----
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/ego_racecar/odom', self._odom_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self._initialpose_cb, 10
        )

        # ---- publishers ----
        self._pose_pub = self.create_publisher(
            Odometry, '/pf/pose/odom', 10
        )
        self._particles_pub = self.create_publisher(PoseArray, '/pf/particles', 10)

        self.get_logger().info('Particle filter node started.')

        # Initialise particles once map arrives (or use /initialpose)
        self._pending_init = True

    # ── parameter helper ──────────────────────────────────────────────────────

    def _params(self):
        class P:
            pass
        p = P()
        p.num_particles      = self.get_parameter('num_particles').value
        p.angle_step         = self.get_parameter('angle_step').value
        p.squash_factor      = self.get_parameter('squash_factor').value
        p.max_range          = self.get_parameter('max_range').value
        p.noise_x            = self.get_parameter('motion_noise_x').value
        p.noise_y            = self.get_parameter('motion_noise_y').value
        p.noise_theta        = self.get_parameter('motion_noise_theta').value
        p.min_particles      = self.get_parameter('min_particles').value
        p.max_particles      = self.get_parameter('max_particles').value
        p.kld_eps            = self.get_parameter('kld_epsilon').value
        p.kld_delta          = self.get_parameter('kld_delta').value
        p.update_min_d       = self.get_parameter('update_min_d').value
        p.update_min_a       = self.get_parameter('update_min_a').value
        p.init_x             = self.get_parameter('initial_pose_x').value
        p.init_y             = self.get_parameter('initial_pose_y').value
        p.init_a             = self.get_parameter('initial_pose_a').value
        p.init_cov_xx        = self.get_parameter('initial_cov_xx').value
        p.init_cov_yy        = self.get_parameter('initial_cov_yy').value
        p.init_cov_aa        = self.get_parameter('initial_cov_aa').value
        return p

    # ── map callback ──────────────────────────────────────────────────────────

    def _map_cb(self, msg: OccupancyGrid):
        self._map = msg
        h, w = msg.info.height, msg.info.width
        raw = np.array(msg.data, dtype=np.int8).reshape((h, w))
        # Convert to float: 0=free, 1=occupied, -1→0 (unknown treated free)
        occ = np.zeros((h, w), dtype=np.float32)
        occ[raw > 0] = raw[raw > 0].astype(np.float32) / 100.0
        self._map_data = occ
        self.get_logger().info(
            f'Map received: {w}x{h} @ {msg.info.resolution:.3f} m/cell'
        )
        if self._pending_init:
            self._init_particles()
            self._pending_init = False

    # ── initialpose callback ───────────────────────────────────────────────────

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        a = _yaw_from_quat(msg.pose.pose.orientation)
        # Re-initialise particle cloud around the provided pose
        p = self._params()
        n = p.num_particles
        self._particles = np.column_stack([
            np.random.normal(x, math.sqrt(p.init_cov_xx), n),
            np.random.normal(y, math.sqrt(p.init_cov_yy), n),
            np.random.normal(a, math.sqrt(p.init_cov_aa), n),
        ])
        self._weights = np.ones(n, dtype=np.float64) / n
        self._last_odom = None
        self._odom_accum_d = 0.0
        self._odom_accum_a = 0.0
        self.get_logger().info(
            f'Reinitialised {n} particles at ({x:.2f}, {y:.2f}, {math.degrees(a):.1f}°)'
        )

    def _init_particles(self):
        p = self._params()
        n = p.num_particles
        self._particles = np.column_stack([
            np.random.normal(p.init_x, math.sqrt(p.init_cov_xx), n),
            np.random.normal(p.init_y, math.sqrt(p.init_cov_yy), n),
            np.random.normal(p.init_a, math.sqrt(p.init_cov_aa), n),
        ])
        self._weights = np.ones(n, dtype=np.float64) / n
        self.get_logger().info(f'Initialised {n} particles.')

    # ── odometry callback: accumulate delta ───────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        a = _yaw_from_quat(msg.pose.pose.orientation)
        curr = np.array([x, y, a])

        self._last_linear_vel = float(msg.twist.twist.linear.x)
        self._last_angular_vel = float(msg.twist.twist.angular.z)

        if self._last_odom is None:
            self._last_odom = curr
            return

        dx = curr[0] - self._last_odom[0]
        dy = curr[1] - self._last_odom[1]
        da = curr[2] - self._last_odom[2]
        # Wrap angle delta
        da = (da + math.pi) % (2 * math.pi) - math.pi

        self._odom_accum_d += math.hypot(dx, dy)
        self._odom_accum_a += abs(da)
        self._last_odom = curr

        if self._particles is None:
            return

        # Always apply motion model to particles (even before scan update)
        self._motion_update(dx, dy, da)

    # ── LiDAR callback: full MCL step ─────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        if self._particles is None or self._map is None:
            return

        p = self._params()

        # Only run correction when robot has moved enough
        if (self._odom_accum_d < p.update_min_d and
                self._odom_accum_a < p.update_min_a):
            return

        self._odom_accum_d = 0.0
        self._odom_accum_a = 0.0

        # Subsample scan beams
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = (msg.angle_min +
                  np.arange(len(ranges)) * msg.angle_increment)
        mask = np.arange(0, len(ranges), p.angle_step)
        ranges = ranges[mask]
        angles = angles[mask]

        # Clip to max_range and remove invalid readings
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < p.max_range)
        ranges = ranges[valid]
        angles = angles[valid]

        if len(ranges) == 0:
            return

        # Correction step
        self._weights = self._compute_weights(ranges, angles, p)

        # Normalise
        w_sum = self._weights.sum()
        if w_sum < 1e-300:
            # All weights collapsed – reinitialise
            self.get_logger().warn('Weight collapse — reinitialising particles.')
            self._init_particles()
            return
        self._weights /= w_sum

        # KLD-adaptive resample
        n_new = self._kld_sample_count(p)
        self._particles, self._weights = self._resample(n_new)

        # Publish
        self._publish_pose(msg.header.stamp)
        self._publish_particles(msg.header.stamp)

    # ── motion model ──────────────────────────────────────────────────────────

    def _motion_update(self, dx: float, dy: float, da: float):
        p = self._params()
        n = len(self._particles)
        noise_x = np.random.normal(0, p.noise_x, n)
        noise_y = np.random.normal(0, p.noise_y, n)
        noise_a = np.random.normal(0, p.noise_theta, n)

        # Rotate displacement into each particle's frame
        cos_a = np.cos(self._particles[:, 2])
        sin_a = np.sin(self._particles[:, 2])
        self._particles[:, 0] += cos_a * dx - sin_a * dy + noise_x
        self._particles[:, 1] += sin_a * dx + cos_a * dy + noise_y
        self._particles[:, 2] += da + noise_a
        # Wrap theta
        self._particles[:, 2] = (self._particles[:, 2] + math.pi) % (2 * math.pi) - math.pi

    # ── sensor model: vectorised ray-casting ──────────────────────────────────

    def _compute_weights(
        self, ranges: np.ndarray, angles: np.ndarray, p
    ) -> np.ndarray:
        """
        Compute particle weights via scan correlation (slides § scan matching).
        For each particle, ray-cast beams into the map occupancy grid, then
        compute Pearson-like correlation between expected and actual ranges.
        """
        map_info = self._map.info
        res = map_info.resolution
        ox = map_info.origin.position.x
        oy = map_info.origin.position.y
        occ = self._map_data  # (H, W)
        H, W = occ.shape

        n_particles = len(self._particles)
        n_beams = len(ranges)
        weights = np.ones(n_particles, dtype=np.float64)

        # Pre-compute beam endpoint offsets for each distance
        # We use a fixed number of ray-march steps
        n_steps = int(p.max_range / res)
        step_r = np.linspace(0, p.max_range, n_steps)  # (S,)

        px = self._particles[:, 0]   # (N,)
        py = self._particles[:, 1]   # (N,)
        pth = self._particles[:, 2]  # (N,)

        for i in range(n_beams):
            beam_angle = pth + angles[i]   # (N,)
            # Ray-march: find expected range for each particle
            cos_b = np.cos(beam_angle)     # (N,)
            sin_b = np.sin(beam_angle)     # (N,)

            expected = np.full(n_particles, p.max_range, dtype=np.float64)

            for s in range(1, n_steps):
                r = step_r[s]
                cx = px + cos_b * r
                cy = py + sin_b * r
                gx = ((cx - ox) / res).astype(np.int32)
                gy = ((cy - oy) / res).astype(np.int32)
                in_bounds = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
                hit = np.zeros(n_particles, dtype=bool)
                hit[in_bounds] = occ[gy[in_bounds], gx[in_bounds]] > 0.5
                newly_hit = hit & (expected == p.max_range)
                expected[newly_hit] = r

            # Likelihood: Gaussian model on range error
            err = ranges[i] - expected
            sigma = 0.5  # metres
            log_like = -0.5 * (err / sigma) ** 2
            weights += log_like / p.squash_factor

        # Convert log-likelihood to weight (subtract max for numerical stability)
        weights -= weights.max()
        return np.exp(weights)

    # ── KLD sample count ──────────────────────────────────────────────────────

    def _kld_sample_count(self, p) -> int:
        """Wilson–Hilferty approximation for KLD-Sampling (Fox 2003)."""
        k = max(2, int(self._weights.size * 0.05))   # rough bin estimate
        if k <= 1:
            return p.num_particles
        z = 2.3263  # 99th percentile of standard normal (delta=0.01)
        n = (k - 1) / (2 * p.kld_eps) * (1 - 2 / (9 * (k - 1)) +
             z * math.sqrt(2 / (9 * (k - 1)))) ** 3
        return int(np.clip(n, p.min_particles, p.max_particles))

    # ── low-variance resampler ─────────────────────────────────────────────────

    def _resample(self, n: int):
        """Systematic (low-variance) resampling."""
        w = self._weights
        cumsum = np.cumsum(w)
        step = 1.0 / n
        u = np.random.uniform(0, step)
        positions = u + step * np.arange(n)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, len(self._particles) - 1)
        new_particles = self._particles[indices].copy()
        new_weights = np.ones(n, dtype=np.float64) / n
        return new_particles, new_weights

    # ── publish best estimate ─────────────────────────────────────────────────

    def _best_pose(self):
        """Weighted mean of particles (circular mean for angle)."""
        w = self._weights
        x = np.average(self._particles[:, 0], weights=w)
        y = np.average(self._particles[:, 1], weights=w)
        sin_a = np.average(np.sin(self._particles[:, 2]), weights=w)
        cos_a = np.average(np.cos(self._particles[:, 2]), weights=w)
        a = math.atan2(sin_a, cos_a)
        return x, y, a

    def _publish_pose(self, stamp):
        x, y, a = self._best_pose()

        # Publish as Odometry so pure pursuit can subscribe directly
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'ego_racecar/base_link'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation = _quat_from_yaw(a)
        msg.twist.twist.linear.x = self._last_linear_vel
        msg.twist.twist.angular.z = self._last_angular_vel
        self._pose_pub.publish(msg)

        # TF: map → odom (correction)
        if self._last_odom is not None:
            ox, oy, oa = self._last_odom
            # The map→odom transform accounts for the difference between
            # the PF estimate in the map frame and raw odom in the odom frame.
            da = a - oa
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = 'map'
            t.child_frame_id = 'odom'
            t.transform.translation.x = x - ox * math.cos(da) + oy * math.sin(da)
            t.transform.translation.y = y - ox * math.sin(da) - oy * math.cos(da)
            t.transform.translation.z = 0.0
            t.transform.rotation = _quat_from_yaw(da)
            self._tf_broadcaster.sendTransform(t)

    def _publish_particles(self, stamp):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        for p in self._particles:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.orientation = _quat_from_yaw(float(p[2]))
            msg.poses.append(pose)
        self._particles_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

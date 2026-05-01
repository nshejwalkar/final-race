# LAB6 Motion Planning Run Instructions

## Overview

This package runs a local motion planner for the F1TENTH platform using **RRT-based planning**.

### Implemented features
- Occupancy-grid generation from `/scan`
- Local goal selection from waypoint CSV
- Online tree expansion in the vehicle frame
- Path extraction and pure-pursuit-style path tracking
- Visualization of:
  - RRT tree
  - Planned path
  - Waypoints
  - Local goal
  - Occupancy grid

## Extra Credit

This implementation includes **RRT\*** for extra credit.

In `rrt_node.py`:

```python
self.use_rrt_star = True
```

RRT\* adds:
- cost function
- neighbor search
- best-parent selection
- rewiring
- cost propagation

This leads to near-optimal paths and satisfies the extra credit requirement.

---

## 1. Setup

```bash
cd ~/roboracer_ws
source /opt/ros/humble/setup.bash
source install/local_setup.bash
```

(Optional rebuild)

```bash
colcon build --symlink-install
source install/local_setup.bash
```

---

## 2. Configuration

Edit `rrt_node.py`:

### Mode switch
```python
self.sim_mode = True   # Simulation
self.sim_mode = False  # Real car
```

### Enable RRT*
```python
self.use_rrt_star = True
```

### Waypoints
```python
self.waypoints_file = ""   # auto load latest CSV
```

---

## 3. Simulation Run

### Terminal 1: Simulator
```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

### Terminal 2: Foxglove
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

### Terminal 3: Planner
```bash
ros2 run lab7_pkg rrt_node
```

---

## 4. Real Car Run

### Step 1: switch mode
```python
self.sim_mode = False
```

### Step 2: make sure running
- particle filter
- /scan
- /pf/pose/odom

### Step 3: run
```bash
ros2 run lab7_pkg rrt_node
```




# RoboRacer Workspace

## Quick Start

Add these aliases to your shell (already in `.bashrc` / `.zshrc`):

```bash
alias setup_fast='source ~/Neel/roboracer_ws/.venv/bin/activate && source /opt/ros/humble/setup.bash && cd ~/Neel/roboracer_ws && ([ ! -d install ] && colcon build); source ~/Neel/roboracer_ws/install/setup.bash'
alias start_sim='ros2 launch f1tenth_gym_ros gym_bridge_launch.py'
```

### Every terminal session

```bash
setup_fast
```

### Run the simulator

```bash
start_sim
```

Open Foxglove at `http://localhost:8765` → import layout from `f1tenth_gym_ros/config/foxglove/gym_bridge_foxglove.json`.

### Run a controller

Wall follower:
```bash
ros2 run wall_follow wall_follow_node
```

Pure pursuit (requires recorded waypoints):
```bash
ros2 launch pure_pursuit pure_pursuit_launch.py
# or with a specific CSV:
ros2 launch pure_pursuit pure_pursuit_launch.py waypoints_path:=/abs/path/to/waypoints.csv
```

Particle filter localizer (run alongside pure pursuit for real-car localization):
```bash
ros2 launch particle_filter localize_launch.py
```

### Record / edit waypoints

```bash
# Edit default race2.csv on my_map1
python waypoint_editor.py

# Create a new waypoints file seeded from the latest existing one
python waypoint_editor.py --csv new_track.csv

# Use a different map
python waypoint_editor.py --csv race2.csv --map f1tenth_gym_ros/maps/my_map1.yaml
```

After saving in the editor, rebuild to update the installed copy:
```bash
colcon build --packages-select pure_pursuit && source install/setup.bash
```

> **Tip:** Run `colcon build --symlink-install` once to skip rebuilds when editing waypoints — the install folder will symlink directly to the source CSVs.

---

## Packages

| Package | Purpose |
|---|---|
| `f1tenth_gym_ros` | Physics simulator bridge + Foxglove visualization |
| `wall_follow` | PID wall-following controller |
| `pure_pursuit` | Waypoint-tracking pure pursuit controller |
| `particle_filter` | MCL localization (real car) |

## Sim config

Edit `f1tenth_gym_ros/config/sim.yaml` to change:
- `map_path` — which map to load
- `num_agent` — 1 or 2 cars
- `sx` / `sy` / `stheta` — ego starting pose
- `sx1` / `sy1` / `stheta1` — opponent starting pose

After any config change: `colcon build --packages-select f1tenth_gym_ros && source install/setup.bash`

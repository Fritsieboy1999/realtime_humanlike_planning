# IIWA Human-like Planning Usage Guide

This guide shows how to use your human-like motion planner with both simulation and real robot.

## Prerequisites for Real Robot

### Network Setup
Configure your PC ethernet connection to connect to KUKA robot:
- **IP Address**: `192.180.1.5`
- **Netmask**: `255.255.255.0` 
- **Gateway**: `192.180.1.1`

Ubuntu: Settings → Network → Wired → Settings → IPv4 → Manual

### Test Connection
```bash
# For Kuka7
ping 192.180.1.7

# For Kuka14  
ping 192.180.1.14
```

## Quick Start

### 1. Setup Environment
```bash
# For simulation
source catkin_ws/src/realtime_humanlike_planning/scripts/setup_network.sh sim

# For real robot  
source catkin_ws/src/realtime_humanlike_planning/scripts/setup_network.sh real
```

After sourcing the setup script, you'll have convenient aliases available:

**Launch Aliases:**
- `launch_sim` - Launch simulation
- `launch_real` - Launch real robot (default Kuka7)
- `launch_kuka7` - Launch Kuka7 robot specifically
- `launch_kuka14` - Launch Kuka14 robot specifically
- `launch_sim_debug` - Launch simulation with debug mode
- `launch_real_debug` - Launch real robot with debug mode

**Connectivity Check Aliases:**
- `ping_kuka7` - Test connection to Kuka7 (192.180.1.7)
- `ping_kuka14` - Test connection to Kuka14 (192.180.1.14)
- `check_joints` - Monitor joint states from robot

### 2. Launch Options

#### **Simulation Mode**
```bash
# Basic simulation with RViz
roslaunch realtime_humanlike_planning iiwa_planning_simulation.launch

# Simulation without RViz
roslaunch realtime_humanlike_planning iiwa_planning_simulation.launch rviz:=false

# Debug mode (better error messages)
roslaunch realtime_humanlike_planning iiwa_planning_simulation.launch debug:=true

# Custom robot IP (not needed for simulation)
roslaunch realtime_humanlike_planning iiwa_planning_simulation.launch robot_name:=iiwa
```

#### **Real Robot Mode**

**IMPORTANT**: Follow this exact sequence for real robot:

1. **Prepare Robot**:
   - Power on robot and smartpad
   - Set smartpad to **AUT** mode (key right → AUT → key left)
   - Alternative: **T1** mode for untested applications

2. **Launch ROS Interface**:
```bash
# For Kuka7 (default)
roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch

# For Kuka14
roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch robot_ip:=192.180.1.14

# Debug mode
roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch debug:=true

# Without safety monitor (not recommended)
roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch safety_monitor:=false
```

3. **Start FRI Application on Smartpad**:
   - Go to **Applications** → **FRIOverlay** 
   - Press **Play ▶️** button
   - Select control mode and stiffness (**<300 recommended**)
   - Keep Application tab **GREEN**

4. **Verify Connection**:
```bash
# Check joint states are publishing
rostopic echo /iiwa/joint_states
```

**Troubleshooting**:
- If connection fails, press **Play ▶️** again on smartpad
- For hard failures: uncheck app in Applications before retrying

## Using the Planner

Once launched, the planning interface provides keyboard commands:

- `p` - Plan new trajectory from current state to goal
- `c` - Confirm current trajectory (shows details & visualization)
- `e` - Execute confirmed trajectory
- `s` - Show current status
- `g` - Set goal position (interactive)
- `w` - Set goal width (interactive)
- `v` - Clear trajectory visualization
- `m` - Move to safe configuration
- `h` - Show help
- `q` - Quit

## Typical Workflow

### Using Aliases (Recommended)
```bash
# 1. Setup environment
source catkin_ws/src/realtime_humanlike_planning/scripts/setup_network.sh real

# 2. Test connectivity (real robot only)
ping_kuka7

# 3. Launch system
launch_kuka7  # or launch_sim for simulation

# 4. Verify connection (real robot only)
check_joints

# 5. Use planner interface...
```

### Step-by-Step Process
1. **Launch the system** (simulation or real robot)
2. **Wait for initialization** (controllers loaded, robot ready)
3. **Set goal position**: Press `g` and enter coordinates
4. **Plan trajectory**: Press `p` 
5. **Confirm trajectory**: Press `c` to see details and visualization
6. **Execute**: Press `e` to run the human-like motion

## Network Configuration

### For Real Robot:
- **PC IP**: `192.180.1.5` (your computer)
- **Kuka7 IP**: `192.180.1.7` (default robot)
- **Kuka14 IP**: `192.180.1.14` (alternative robot)
- **ROS Master**: Runs on your PC (`192.180.1.5:11311`)
- **FRI Connection**: Requires FRIOverlay application on smartpad

### For Simulation:
- **Local ROS Master**: No network configuration needed
- **Gazebo**: Runs locally with physics simulation

## Shutdown Procedure (Real Robot)

1. **Stop Application**: Stop any running applications on smartpad
2. **Shutdown Smartpad**: Applications → shutdown → run
3. **Power Off Robot**: Press power button on robot base

## Troubleshooting

### Common Issues:
1. **"FRI not found"** → Check `FRI_DIR` environment variable
2. **"Controller not ready"** → Wait longer for robot initialization
3. **"Safety violation"** → Check joint/velocity limits in safety monitor
4. **"Planning failed"** → Check robot configuration and goal reachability

### Debug Mode:
Use `debug:=true` for better error messages and Python environment handling.

## File Structure

- `iiwa_planning_simulation.launch` - Simulation setup
- `iiwa_planning_real_robot.launch` - Real robot setup  
- `setup_network.sh` - Network configuration helper
- `iiwa_planning_interface.py` - Main planning interface
- `safety_monitor.py` - Safety monitoring for real robot

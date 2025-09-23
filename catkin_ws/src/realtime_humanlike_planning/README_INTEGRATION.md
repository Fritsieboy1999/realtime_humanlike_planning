# IIWA Planning Interface

A comprehensive ROS integration that combines:
- Gazebo simulation and real IIWA robot support
- Cartesian trajectory impedance controller from iiwa_impedance_control
- Real-time human-like trajectory planning
- Interactive keyboard control interface
- Trajectory visualization in RViz

## Quick Start

### 1. Simulation Mode

```bash
# Start the complete simulation environment
roslaunch realtime_humanlike_planning iiwa_planning_simulation.launch

# The interface will start with keyboard commands available:
# p - Plan new trajectory from current state to goal
# c - Confirm current trajectory for execution  
# e - Execute confirmed trajectory
# s - Show current status
# g - Set goal position (interactive)
# w - Set goal width (interactive)
# h - Show help
# q - Quit
```

### 2. Real Robot Mode

```bash
# Connect to real IIWA robot (adjust IP as needed)
roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch robot_ip:=172.31.1.147

# Same keyboard interface as simulation
```

## Features

### Planning Interface
- **Real-time Planning**: Uses optimized human-like motion planner with 30-step horizon
- **Warm Starting**: Subsequent trajectories benefit from previous solutions
- **Fitts' Law Integration**: Realistic movement timing based on task difficulty
- **Safety Monitoring**: Continuous monitoring of joint velocities and workspace limits

### Visualization
- **Trajectory Display**: Green line showing planned end-effector path
- **Goal Visualization**: Yellow sphere showing target position with tolerance
- **Start/End Points**: Blue (start) and red (end) markers
- **Robot Model**: Full IIWA robot visualization in RViz

### Control Interface
- **Interactive Commands**: Single-key commands for all operations
- **Goal Setting**: Interactive goal position and tolerance setting
- **Status Monitoring**: Real-time robot state and trajectory information
- **Confirmation System**: Two-step process (plan → confirm → execute) for safety

## Workflow

1. **Start System**: Launch appropriate launch file for simulation or real robot
2. **Set Goal**: Use 'g' command to set desired end-effector position
3. **Plan Trajectory**: Press 'p' to generate human-like trajectory
4. **Review**: Trajectory is visualized in RViz for review
5. **Confirm**: Press 'c' to confirm trajectory for execution
6. **Execute**: Press 'e' to send trajectory to impedance controller
7. **Monitor**: Watch execution in RViz and check status with 's'

## Architecture

### Main Components
- `iiwa_planning_interface.py`: Main integration node
- `safety_monitor.py`: Safety monitoring and emergency stop
- Launch files for simulation and real robot modes
- RViz configuration for visualization

### Integration Points
- **Gazebo**: Full physics simulation with IIWA robot
- **iiwa_impedance_control**: Cartesian trajectory execution
- **realtime_humanlike_planning**: Human-like motion generation
- **RViz**: Visualization and monitoring

## Dependencies

- ROS (tested with Melodic/Noetic)
- Gazebo
- iiwa_ros package suite
- iiwa_impedance_control
- Python packages: numpy, casadi, matplotlib

## Safety Features

- Joint velocity monitoring
- Workspace boundary checking
- Communication timeout detection
- Emergency stop capability
- Controller management integration

## Customization

### Goal Parameters
- Default goal: [0.5, 0.2, 0.6] meters
- Default tolerance: 0.015 meters
- Adjustable via interactive commands

### Planning Parameters
- Horizon: 30 steps (adjustable in code)
- Solver: MUMPS for smooth trajectories
- Fitts' Law: Enabled for realistic timing

## Troubleshooting

### Common Issues
1. **Action server not available**: Ensure impedance controller is loaded
2. **No joint state data**: Check robot connection and joint_state_publisher
3. **Planning fails**: Verify goal is reachable and robot state is valid
4. **Visualization not showing**: Check RViz topics and frame configurations

### Debug Commands
```bash
# Check robot state
rostopic echo /iiwa/joint_states

# Check action server
rostopic list | grep CartesianTrajectoryExecution

# Monitor safety
rostopic echo /iiwa_safety_monitor/safety_status
```

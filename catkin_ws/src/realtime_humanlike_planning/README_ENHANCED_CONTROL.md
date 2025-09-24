# Enhanced IIWA Planning Interface 🚀

This document describes the **reconstructed and enhanced** IIWA planning system with dual-mode capabilities, streamlined configuration, and comprehensive visualization.

## 🎯 **System Overview**

The enhanced system provides:
- **Dual Planning Modes**: Goal planning and Z-plane targeting
- **Centralized Configuration**: Single YAML file for all settings
- **Enhanced Controller Management**: Automatic switching between control modes
- **Rich Visualization**: Real-time trajectory, goal, and z-plane visualization
- **Initial Trajectory Visualization**: Shows straight-line path before optimization

## 🔧 **Components**

### 1. Enhanced Planning Interface
**File**: `iiwa_planning_interface_enhanced.py`
- Two planning modes: goal position and z-plane targeting
- Interactive parameter adjustment (goal, z-plane height, target width)
- Real-time visualization of targets and trajectories
- Impedance stiffness parameter updating
- Keyboard-based interface with comprehensive help system

### 2. Enhanced Controller Manager
**File**: `enhanced_controller_manager.py`
- **Simulation Mode**: Uses only CartesianImpedanceController
- **Real Robot Mode**: Supports both hand-guiding and impedance control
- Automatic controller switching based on environment
- Dynamic impedance parameter configuration

### 3. Centralized Configuration
**File**: `config/planning_config.yaml`
- Initial robot joint configuration
- Cartesian impedance stiffness and damping parameters
- Planning parameters (horizon, solver)
- Visualization settings
- Default goal positions and z-plane heights

### 4. Enhanced Launch File
**File**: `launch/iiwa_planning_enhanced_simulation.launch`
- Launches Gazebo with specified initial joint configuration
- Loads all required controllers
- Starts enhanced controller manager and planning interface
- Optional RViz launch with custom configuration

## 🎮 **Usage**

### Launching the System
```bash
# For simulation
roslaunch realtime_humanlike_planning iiwa_planning_enhanced_simulation.launch

# With GUI (Gazebo visual interface)
roslaunch realtime_humanlike_planning iiwa_planning_enhanced_simulation.launch gui:=true

# Without RViz
roslaunch realtime_humanlike_planning iiwa_planning_enhanced_simulation.launch rviz:=false

# Debug mode (shows Python output)
roslaunch realtime_humanlike_planning iiwa_planning_enhanced_simulation.launch debug:=true
```

### Interactive Interface Commands

#### **Planning Mode Selection**
- `1` - Switch to goal planning mode
- `2` - Switch to z-plane planning mode

#### **Goal Planning Mode**
- `g` - Set goal position (X, Y, Z coordinates)
- `w` - Set target width
- `p` - Plan humanlike trajectory to goal
- `c` - Confirm planned trajectory
- `e` - Execute confirmed trajectory

#### **Z-Plane Planning Mode**
- `z` - Set z-plane height
- `w` - Set target width  
- `p` - Plan humanlike trajectory to z-plane
- `c` - Confirm planned trajectory
- `e` - Execute confirmed trajectory

#### **General Commands**
- `k` - Update Cartesian impedance stiffness parameters
- `v` - Clear and refresh visualizations
- `s` - Show current system status
- `h` - Show help menu
- `q` - Return to planning options or quit

## 📊 **Configuration Settings**

### Robot Configuration
```yaml
# Initial joint positions [rad]
initial_joint_state: [0.0, 0.280260657, 0.0, -1.53136477, 0.0, 1.31590271, 0.0]

# Cartesian impedance parameters [N/m, Nm/rad]
initial_cartesian_stiffness: [0.1, 0.1, 300, 300, 300, 300]
initial_cartesian_damping: [2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
```

### Planning Parameters
```yaml
planning:
  horizon: 30          # Planning horizon steps
  solver_type: "mumps" # Optimization solver

# Initial target settings
initial_target_width: 0.015           # meters
default_goal_position: [0.5, 0.2, 0.6] # X, Y, Z
default_z_plane_height: 0.6           # meters
```

### Visualization Settings
```yaml
visualization:
  z_plane_size: 1.0                    # Size of z-plane visualization
  z_plane_alpha: 0.3                   # Transparency of z-plane
  target_scale: 0.05                   # Scaling for target markers
  show_initial_trajectory: true        # Show straight-line initial guess
  initial_trajectory_alpha: 0.6        # Transparency of initial trajectory
```

## 🎨 **Visualization Features**

### Goal Planning Mode
- **Red sphere**: Goal position marker
- **Transparent red sphere**: Target width visualization (2x width diameter)
- **Yellow line**: Initial trajectory (straight-line interpolation)
- **Green line**: Planned humanlike trajectory (after pressing 'p')

### Z-Plane Planning Mode
- **Blue plane**: Target z-plane (transparent)
- **Orange sphere**: Projected goal on z-plane
- **Transparent orange sphere**: Target width on z-plane
- **Yellow line**: Initial trajectory to z-plane
- **Green line**: Planned humanlike trajectory

## 🔄 **Control Modes**

### Simulation
- **Primary**: CartesianImpedanceController (for trajectory execution)
- **Secondary**: joint_state_controller (required for impedance control)

### Real Robot
- **Hand-guiding Mode**: Native KUKA hand-guiding with torque control
- **Impedance Mode**: CartesianImpedanceController for trajectory execution
- **Automatic Switching**: System switches between modes as needed

## 🚀 **Workflow**

1. **Launch System**: Use the enhanced launch file
2. **Select Planning Mode**: Press '1' for goal or '2' for z-plane
3. **Adjust Parameters**: Set goal position, z-height, or target width
4. **Plan**: Press 'p' to compute humanlike trajectory
5. **Confirm**: Press 'c' to confirm the planned trajectory
6. **Execute**: Press 'e' to send trajectory to robot
7. **Adjust Stiffness**: Press 'k' anytime to modify impedance parameters

## 📝 **Key Features Restored**

✅ **Z-plane targeting functionality**  
✅ **Initial trajectory visualization**  
✅ **Enhanced controller management**  
✅ **Centralized configuration**  
✅ **Dual planning modes**  
✅ **Rich interactive interface**  
✅ **Real-time parameter adjustment**  
✅ **Comprehensive visualization**

## 🛠️ **Technical Notes**

- The system automatically switches to impedance control mode for trajectory execution
- For real robot: automatically returns to hand-guiding mode after execution
- All configuration is loaded from `config/planning_config.yaml`
- Visualization updates every 2 seconds to maintain real-time feedback
- Controller switching includes proper timing delays for stable operation

## 🔍 **Troubleshooting**

If you encounter issues:
1. Ensure all controllers are properly loaded (`rosservice call /iiwa/controller_manager/list_controllers`)
2. Check that the planning configuration file is properly loaded
3. Verify RViz is configured to display the correct marker topics
4. Use debug mode (`debug:=true`) to see detailed Python output

The enhanced system combines the simplicity you requested with the full functionality of z-plane targeting and visualization!

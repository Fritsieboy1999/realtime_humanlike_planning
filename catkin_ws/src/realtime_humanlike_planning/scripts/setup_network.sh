#!/bin/bash
# Network configuration script for IIWA robot
# Usage: source setup_network.sh [sim|real]

MODE=${1:-sim}

# Common ROS setup
source /opt/ros/noetic/setup.bash
source /home/fvanhall/Desktop/realtime_humanlike_planning/catkin_ws/devel/setup.bash

# Set FRI library path for real robot support
export FRI_DIR=/home/fvanhall/Desktop/realtime_humanlike_planning/kuka_dependencies/kuka_fri

if [ "$MODE" = "real" ]; then
    echo "🤖 Setting up REAL ROBOT network configuration"
    
    # KUKA Robot network configuration (per lab instructions)
    export ROS_MASTER_URI=http://192.180.1.5:11311/
    export ROS_IP=192.180.1.5
    
    echo "✅ Robot network configured:"
    echo "   ROS_MASTER_URI: $ROS_MASTER_URI"  
    echo "   ROS_IP: $ROS_IP"
    echo ""
    echo "📋 BEFORE launching, ensure:"
    echo "   1. Robot is powered on and in AUT mode"
    echo "   2. PC ethernet configured to 192.180.1.5/255.255.255.0"
    echo "   3. Robot is reachable:"
    echo "      - Kuka7: ping 192.180.1.7"
    echo "      - Kuka14: ping 192.180.1.14"
    echo ""
    echo "🔄 AFTER launching, you must:"
    echo "   1. Start FRIOverlay application on smartpad"
    echo "   2. Select control mode and stiffness (<300)"
    echo "   3. Check joint_states: rostopic echo /iiwa/joint_states"
    
elif [ "$MODE" = "sim" ]; then
    echo "🎮 Setting up SIMULATION network configuration"
    
    # Simulation mode - local ROS master
    unset ROS_MASTER_URI
    unset ROS_IP
    
    echo "✅ Local simulation network configured"
    
else
    echo "❌ Invalid mode: $MODE"
    echo "Usage: source setup_network.sh [sim|real]"
    return 1
fi

# Helpful aliases
alias launch_sim="roslaunch realtime_humanlike_planning iiwa_planning_simulation.launch"
alias launch_real="roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch"
alias launch_kuka7="roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch robot_ip:=192.180.1.7"
alias launch_kuka14="roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch robot_ip:=192.180.1.14"
alias launch_sim_debug="roslaunch realtime_humanlike_planning iiwa_planning_simulation.launch debug:=true"
alias launch_real_debug="roslaunch realtime_humanlike_planning iiwa_planning_real_robot.launch debug:=true"

# Robot connectivity check aliases
alias ping_kuka7="ping 192.180.1.7"
alias ping_kuka14="ping 192.180.1.14"
alias check_joints="rostopic echo /iiwa/joint_states"

echo "🚀 Environment ready! Available commands:"
echo "   launch_sim        - Launch simulation"
echo "   launch_real       - Launch real robot interface (default Kuka7)"
echo "   launch_kuka7      - Launch Kuka7 robot specifically"
echo "   launch_kuka14     - Launch Kuka14 robot specifically"
echo "   launch_sim_debug  - Launch simulation with debug mode"
echo "   launch_real_debug - Launch real robot with debug mode"
echo ""
echo "🔍 Robot connectivity checks:"
echo "   ping_kuka7        - Test connection to Kuka7"
echo "   ping_kuka14       - Test connection to Kuka14" 
echo "   check_joints      - Monitor joint states from robot"

#!/bin/bash
# Simple wrapper to quickly clean up Gazebo processes
# Usage: ./kill_gazebo.sh

echo "🛑 Quick Gazebo cleanup..."

# Kill the main processes
pkill -f gzserver
pkill -f gzclient  
pkill -f gazebo

# Kill any hanging ROS launch processes
pkill -f "roslaunch.*iiwa.*simulation"

# Give them a moment to shut down
sleep 2

echo "✅ Quick cleanup done!"

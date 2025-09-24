#!/bin/bash

# Comprehensive Gazebo Cleanup Script
# This script ensures clean shutdown of all Gazebo-related processes and ROS nodes

echo "🧹 Starting Gazebo cleanup..."

# Function to kill processes by name
kill_processes() {
    local process_name=$1
    local pids=$(pgrep -f "$process_name")
    if [ ! -z "$pids" ]; then
        echo "🔪 Killing $process_name processes: $pids"
        echo $pids | xargs kill -TERM 2>/dev/null
        sleep 2
        # Force kill if still running
        local remaining_pids=$(pgrep -f "$process_name")
        if [ ! -z "$remaining_pids" ]; then
            echo "💀 Force killing remaining $process_name processes: $remaining_pids"
            echo $remaining_pids | xargs kill -KILL 2>/dev/null
        fi
    else
        echo "✅ No $process_name processes found"
    fi
}

# Function to clean up ROS nodes
cleanup_ros_nodes() {
    echo "🛑 Stopping ROS nodes..."
    
    # Kill specific nodes that might be hanging
    rosnode kill /gazebo 2>/dev/null || true
    rosnode kill /robot_state_publisher 2>/dev/null || true
    rosnode kill /iiwa/iiwa_service 2>/dev/null || true
    rosnode kill /enhanced_controller_manager 2>/dev/null || true
    rosnode kill /iiwa_planning_interface_enhanced 2>/dev/null || true
    
    sleep 2
    
    # List any remaining nodes
    echo "📋 Remaining ROS nodes:"
    rosnode list 2>/dev/null | grep -v "^/rosout$" | head -10 || echo "No additional nodes found"
}

# Kill Gazebo processes
kill_processes "gzserver"
kill_processes "gzclient" 
kill_processes "gazebo"

# Kill ROS launch processes
kill_processes "roslaunch.*iiwa.*simulation"
kill_processes "roslaunch.*gazebo"

# Clean up ROS nodes if ROS master is running
if pgrep -f "rosmaster" > /dev/null; then
    echo "🤖 ROS master detected, cleaning up nodes..."
    cleanup_ros_nodes
else
    echo "ℹ️  No ROS master running"
fi

# Clean up any remaining controller or planning processes
kill_processes "enhanced_controller_manager"
kill_processes "iiwa_planning_interface"
kill_processes "iiwa_service"

# Clean up shared memory and semaphores that Gazebo might have left behind
echo "🧹 Cleaning up shared memory..."
ipcs -s | grep $USER | awk '{print $2}' | xargs -r ipcrm sem 2>/dev/null || true
ipcs -m | grep $USER | awk '{print $2}' | xargs -r ipcrm shm 2>/dev/null || true

# Remove Gazebo temporary files
echo "📁 Cleaning up temporary files..."
rm -rf /tmp/gazebo-$USER-* 2>/dev/null || true
rm -rf ~/.gazebo/log/* 2>/dev/null || true

echo "✅ Gazebo cleanup completed!"

# Wait a moment before allowing relaunch
echo "⏱️  Waiting 2 seconds for complete cleanup..."
sleep 2

echo "🚀 Ready to relaunch!"

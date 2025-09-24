#!/usr/bin/env python3
"""
Integrated IIWA Planning Interface Node

This node provides a comprehensive interface that integrates:
- Gazebo simulation and real robot switching
- Cartesian trajectory impedance controller from iiwa_impedance_control
- Real-time human-like trajectory planning using realtime_humanlike_planning
- Robot state reading and monitoring
- Interactive keyboard commands for trajectory recalculation and execution
- Trajectory visualization in RViz/Gazebo

Author: Assistant
"""

import rospy
import numpy as np
import threading
import time
from typing import Optional, Dict, Any, List
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import actionlib
from iiwa_impedance_control.msg import JointTrajectoryExecutionAction, JointTrajectoryExecutionGoal, CartesianTrajectoryExecutionAction, CartesianTrajectoryExecutionGoal
from std_msgs.msg import Float64MultiArray
import sys
import termios
import select
import os

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add ROS site-packages to Python path for pinocchio
ros_site_packages = '/opt/ros/noetic/lib/python3.8/site-packages'
if ros_site_packages not in sys.path:
    sys.path.insert(0, ros_site_packages)

# Import the planning modules
from planning import VanHallHumanReaching3D_Optimized
from parameters.tasks.default_task_3d import TaskParams3D
from parameters.rewards.van_hall_reward_3d import VanHallRewardParams3D


class IIWAPlanningInterface:
    """
    Main interface node that orchestrates all components for human-like trajectory planning and execution.
    """
    
    def __init__(self):
        """Initialize the planning interface node."""
        rospy.init_node('iiwa_planning_interface', anonymous=True)
        
        # Configuration
        self.robot_name = rospy.get_param('~robot_name', 'iiwa')
        self.use_simulation = rospy.get_param('~use_simulation', True)
        self.controller_type = rospy.get_param('~controller_type', 'CartesianImpedanceController')
        
        # Planning parameters
        self.goal_position = np.array([0.5, 0.2, 0.6])  # Default goal position
        self.goal_width = 0.015  # Default goal width
        
        # Default safe joint configuration (not all zeros)
        self.safe_joint_config = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
        
        # State variables
        self.current_joint_state = None
        self.current_ee_pose = None
        self.last_planned_trajectory = None
        self.trajectory_confirmed = False
        
        # Initialize components in order
        self._init_tf()
        self._init_publishers()
        self._init_planner()  # Initialize planner before subscribers
        self._init_subscribers()
        self._init_action_clients()
        
        # Keyboard input handling
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        rospy.loginfo("IIWA Planning Interface initialized successfully!")
        rospy.loginfo(f"Mode: {'Simulation' if self.use_simulation else 'Real Robot'}")
        rospy.loginfo(f"Controller: {self.controller_type}")
        
    def _init_tf(self):
        """Initialize TF2 components."""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        
    def _init_publishers(self):
        """Initialize ROS publishers."""
        self.trajectory_marker_pub = rospy.Publisher(
            '/iiwa_planning_interface/trajectory_markers', 
            MarkerArray, 
            queue_size=10
        )
        
        self.goal_marker_pub = rospy.Publisher(
            '/iiwa_planning_interface/goal_marker',
            Marker,
            queue_size=10
        )
        
    def _init_subscribers(self):
        """Initialize ROS subscribers."""
        joint_states_topic = f'/{self.robot_name}/joint_states'
        self.joint_state_sub = rospy.Subscriber(
            joint_states_topic,
            JointState,
            self._joint_state_callback
        )
        
    def _init_action_clients(self):
        """Initialize action clients for trajectory execution."""
        cartesian_action_name = f'/{self.robot_name}/CartesianImpedanceController/cartesian_trajectory_execution_action'
        self.cartesian_action_client = actionlib.SimpleActionClient(
            cartesian_action_name,
            CartesianTrajectoryExecutionAction
        )
        
        rospy.loginfo(f"Waiting for action server: {cartesian_action_name}")
        if not self.cartesian_action_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logwarn(f"Action server {cartesian_action_name} not available. Trajectory execution will be disabled.")
            self.cartesian_action_client = None
        else:
            rospy.loginfo("Cartesian trajectory action server connected!")
            
    def _init_planner(self):
        """Initialize the human-like motion planner."""
        try:
            # Create reward parameters with default cost combination
            reward_params = VanHallRewardParams3D.default()
            reward_params.use_fitts_law = True  # Enable Fitts' law for realistic timing
            
            # Initialize planner with warm start disabled for better trajectory continuity
            self.planner = VanHallHumanReaching3D_Optimized(
                H=30,  # Reduced horizon for real-time performance
                reward_params=reward_params,
                solver_type="mumps"  # Use MUMPS for smooth trajectories
            )
            
            rospy.loginfo("Human-like motion planner initialized successfully!")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize planner: {e}")
            self.planner = None
    
    def _joint_state_callback(self, msg: JointState):
        """Callback for joint state updates."""
        self.current_joint_state = msg
        
        # Don't update current_ee_pose here - it should only be updated with controller's actual pose
        # during trajectory planning to ensure accuracy and prevent overwriting the correct position
        pass
    
    def _extract_joint_positions(self, joint_state_msg: JointState) -> Optional[np.ndarray]:
        """Extract joint positions in the correct order for the 7-DOF IIWA."""
        expected_joints = [
            'iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4',
            'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7'
        ]
        
        try:
            joint_positions = np.zeros(7)
            for i, joint_name in enumerate(expected_joints):
                if joint_name in joint_state_msg.name:
                    idx = joint_state_msg.name.index(joint_name)
                    joint_positions[i] = joint_state_msg.position[idx]
                else:
                    rospy.logwarn(f"Joint {joint_name} not found in joint state message")
                    return None
            
            return joint_positions
            
        except Exception as e:
            rospy.logwarn(f"Failed to extract joint positions: {e}")
            return None
    
    def get_current_robot_state(self) -> Optional[np.ndarray]:
        """Get the current robot state [q1...q7, dq1...dq7]."""
        if self.current_joint_state is None:
            rospy.logwarn("No joint state information available")
            return None
        
        joint_positions = self._extract_joint_positions(self.current_joint_state)
        if joint_positions is None:
            return None
        
        # Extract joint velocities in the same order as positions
        expected_joints = [
            'iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4',
            'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7'
        ]
        
        joint_velocities = np.zeros(7)
        if len(self.current_joint_state.velocity) > 0:
            for i, joint_name in enumerate(expected_joints):
                if joint_name in self.current_joint_state.name:
                    idx = self.current_joint_state.name.index(joint_name)
                    if idx < len(self.current_joint_state.velocity):
                        joint_velocities[i] = self.current_joint_state.velocity[idx]
        
        # Combine positions and velocities - ensure exactly 14 elements
        robot_state = np.concatenate([joint_positions, joint_velocities])
        
        # Debug logging
        rospy.logdebug(f"Robot state size: {robot_state.shape}, positions: {joint_positions.shape}, velocities: {joint_velocities.shape}")
        
        if robot_state.shape[0] != 14:
            rospy.logerr(f"Invalid robot state size: {robot_state.shape[0]}, expected 14")
            return None
            
        return robot_state
    
    def plan_trajectory(self) -> bool:
        """Plan a new trajectory from current state to goal."""
        if self.planner is None:
            rospy.logerr("Planner not initialized")
            return False
        
        # Wait for fresh joint state update to ensure we have the latest robot position
        rospy.loginfo("Waiting for fresh robot state...")
        last_joint_state = self.current_joint_state
        timeout = rospy.Time.now() + rospy.Duration(2.0)  # 2 second timeout
        
        while rospy.Time.now() < timeout:
            rospy.sleep(0.1)
            if self.current_joint_state is not None and self.current_joint_state != last_joint_state:
                break
        else:
            rospy.logwarn("No fresh joint state received, using last available state")
        
        # Get current robot state
        current_state = self.get_current_robot_state()
        if current_state is None:
            rospy.logerr("Cannot get current robot state")
            return False
        
        # Get fresh current end-effector position from controller to ensure accuracy
        try:
            controller_pose_msg = rospy.wait_for_message('/CartesianImpedanceController/cartesian_pose', PoseStamped, timeout=1.0)
            controller_ee_pos = np.array([
                controller_pose_msg.pose.position.x,
                controller_pose_msg.pose.position.y,
                controller_pose_msg.pose.position.z
            ])
            self.current_ee_pose = controller_ee_pos
            rospy.loginfo(f"✅ Updated current EE position from controller: {controller_ee_pos}")
        except Exception as e:
            rospy.logwarn(f"Could not get controller pose, using computed position: {e}")
        
        rospy.loginfo(f"Current state size: {current_state.shape}")
        rospy.loginfo(f"Joint positions: {current_state[:7]}")
        rospy.loginfo(f"Joint velocities: {current_state[7:]}")
        
        # Check if robot is in a singular configuration (all joints near zero)
        if np.allclose(current_state[:7], 0.0, atol=0.1):
            rospy.logwarn("Robot is near zero configuration, which may cause kinematics issues")
            rospy.loginfo("Consider moving robot to a safer configuration first")
        
        rospy.loginfo("Planning new trajectory...")
        rospy.loginfo(f"  From EE: {self.current_ee_pose}")
        rospy.loginfo(f"  To goal: {self.goal_position}")
        rospy.loginfo(f"  Goal width: {self.goal_width}")
        
        # Calculate distance for better planning
        if self.current_ee_pose is not None:
            distance = np.linalg.norm(self.goal_position - self.current_ee_pose)
            rospy.loginfo(f"  Distance: {distance:.3f}m")
        
        try:
            # Create task - split current_state into positions and velocities
            q_start = current_state[:7]   # First 7 elements are positions
            dq_start = current_state[7:]  # Last 7 elements are velocities
            
            # Use the actual current joint state from ROS topic - simple and direct
            rospy.loginfo(f"✅ Using current joint state from /iiwa/joint_states")
            rospy.loginfo(f"   Joint positions: {q_start}")
            rospy.loginfo(f"   Joint velocities: {dq_start}")
            
            # Transform goal from controller frame to planner frame
            goal_planner_frame = self.planner.kinematics.transform_from_controller_frame(self.goal_position)
            rospy.loginfo(f"   Goal in controller frame: {self.goal_position}")
            rospy.loginfo(f"   Goal in planner frame: {goal_planner_frame}")
            
            task = TaskParams3D.create_reaching_task(
                q_start=q_start,
                dq_start=dq_start,
                goal_position=goal_planner_frame,  # Use transformed goal
                width=self.goal_width
            )
            
            # Debug: Check what the planner actually receives
            rospy.loginfo(f"🔍 Debug - Task created:")
            rospy.loginfo(f"   xi0 (first 7): {task.xi0[:7]}")
            rospy.loginfo(f"   xi0 (last 7): {task.xi0[7:]}")
            rospy.loginfo(f"   goal: {task.goal}")
            
            # Verify the starting EE position that the planner will compute
            if self.planner and self.planner.kinematics:
                try:
                    # Use same FK as planner optimization for consistency
                    planner_start_ee_internal = self.planner.kinematics.fwd_kin(task.xi0[:7])
                    planner_start_ee_internal = np.array(planner_start_ee_internal).flatten()
                    
                    # Transform to controller frame for comparison
                    planner_start_ee = self.planner.kinematics.transform_to_controller_frame(planner_start_ee_internal)
                    
                    rospy.loginfo(f"   Planner will compute start EE as: {planner_start_ee} (controller frame)")
                    rospy.loginfo(f"   Controller actual EE: {self.current_ee_pose}")
                    if self.current_ee_pose is not None:
                        error = np.linalg.norm(planner_start_ee - self.current_ee_pose)
                        rospy.loginfo(f"   Error: {error*1000:.1f}mm")
                except Exception as e:
                    rospy.logwarn(f"Could not compute planner start EE: {e}")
            
            # Task metadata for Fitts' law
            if self.current_ee_pose is not None:
                distance = np.linalg.norm(self.goal_position - self.current_ee_pose)
            else:
                distance = 0.5  # Default distance
            
            task_meta = {
                'width': self.goal_width,
                'distance': distance,
                'task_number': 1
            }
            
            # Solve trajectory
            start_time = time.time()
            result = self.planner.solve(task, task_meta, start_ee_pos=self.current_ee_pose)
            solve_time = time.time() - start_time
            
            # No corrections needed - trajectory should already start from current state
            
            # Store result
            self.last_planned_trajectory = result
            self.trajectory_confirmed = False
            
            # Log results
            H = result.t.shape[0]
            total_time = result.t[-1]
            final_error = np.linalg.norm(result.ee_positions[:, -1] - self.goal_position)
            
            rospy.loginfo(f"✅ Trajectory planned successfully! ({solve_time:.3f}s)")
            rospy.loginfo(f"  Horizon: {H} steps, Duration: {total_time:.3f}s")
            rospy.loginfo(f"  Final error: {final_error*1000:.2f}mm")
            
            # Visualize trajectory
            self._visualize_trajectory(result)
            self._visualize_goal()
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Trajectory planning failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _reconnect_action_client(self):
        """Try to reconnect to the action server."""
        cartesian_action_name = f'/{self.robot_name}/CartesianImpedanceController/cartesian_trajectory_execution_action'
        
        if self.cartesian_action_client is None:
            rospy.loginfo(f"Attempting to reconnect to action server: {cartesian_action_name}")
            self.cartesian_action_client = actionlib.SimpleActionClient(
                cartesian_action_name,
                CartesianTrajectoryExecutionAction
            )
        
        # Try to connect with a shorter timeout
        if self.cartesian_action_client.wait_for_server(timeout=rospy.Duration(3.0)):
            rospy.loginfo("✅ Action server connected!")
            return True
        else:
            rospy.logwarn("❌ Action server still not available")
            self.cartesian_action_client = None
            return False

    def execute_trajectory(self) -> bool:
        """Execute the last planned trajectory using real-time reference pose publishing."""
        if self.last_planned_trajectory is None:
            rospy.logerr("No trajectory to execute. Plan a trajectory first.")
            return False
        
        if not self.trajectory_confirmed:
            rospy.logwarn("Trajectory not confirmed. Confirm with 'c' command first.")
            return False
        
        rospy.loginfo("Executing human-like trajectory with real-time reference poses...")
        
        try:
            trajectory = self.last_planned_trajectory
            
            # Create publisher for reference poses (now with working subscriber!)
            ref_pose_pub = rospy.Publisher(
                '/iiwa/CartesianImpedanceController/reference_pose', 
                PoseStamped, 
                queue_size=10
            )
            
            # Wait for publisher to be ready
            rospy.sleep(0.5)
            
            # Execute trajectory by publishing reference poses at high frequency
            rate = rospy.Rate(100)  # 100 Hz for smooth impedance control
            start_time = rospy.Time.now()
            
            rospy.loginfo(f"Publishing complete human-like trajectory with {trajectory.q.shape[1]} waypoints at 100Hz")
            rospy.loginfo(f"Trajectory duration: {trajectory.t[-1]:.3f}s")
            
            for i in range(trajectory.q.shape[1]):
                if rospy.is_shutdown():
                    break
                
                # Get current time in trajectory
                elapsed_time = (rospy.Time.now() - start_time).to_sec()
                
                # Skip if we're ahead of the trajectory timing (maintain proper timing)
                if elapsed_time < trajectory.t[i]:
                    rate.sleep()
                    continue
                
                # Compute end-effector pose for this waypoint
                if hasattr(self.planner, 'kinematics') and self.planner.kinematics:
                    try:
                        # Use same FK as planner for consistency
                        ee_pos_internal = self.planner.kinematics.fwd_kin(trajectory.q[:, i])
                        ee_pos_internal = np.array(ee_pos_internal).flatten()
                        
                        # Transform to controller frame for sending to controller
                        ee_pos = self.planner.kinematics.transform_to_controller_frame(ee_pos_internal)
                        
                        # Create reference pose message
                        ref_pose = PoseStamped()
                        ref_pose.header.frame_id = "iiwa_link_0"
                        ref_pose.header.stamp = rospy.Time.now()
                        
                        ref_pose.pose.position.x = ee_pos[0]
                        ref_pose.pose.position.y = ee_pos[1]
                        ref_pose.pose.position.z = ee_pos[2]
                        
                        # End-effector facing down (180° rotation around X-axis)
                        ref_pose.pose.orientation.w = 0.0
                        ref_pose.pose.orientation.x = 1.0
                        ref_pose.pose.orientation.y = 0.0
                        ref_pose.pose.orientation.z = 0.0
                        
                        # Publish reference pose at high frequency
                        ref_pose_pub.publish(ref_pose)
                        
                        if i % 50 == 0:  # Log every 50th waypoint
                            rospy.loginfo(f"Publishing waypoint {i+1}/{trajectory.q.shape[1]} at t={trajectory.t[i]:.3f}s")
                        
                    except Exception as e:
                        rospy.logwarn(f"Failed to compute FK for waypoint {i}: {e}")
                
                rate.sleep()
            
            rospy.loginfo("✅ Complete human-like trajectory published successfully!")
            
            # Wait for the robot to settle at the final position
            rospy.loginfo("⏳ Waiting 2.0s for robot to settle at final position...")
            rospy.sleep(2.0)
            
            # Verify final position accuracy using controller's pose
            try:
                final_controller_pose = rospy.wait_for_message('/CartesianImpedanceController/cartesian_pose', PoseStamped, timeout=1.0)
                final_ee_pos = np.array([
                    final_controller_pose.pose.position.x,
                    final_controller_pose.pose.position.y,
                    final_controller_pose.pose.position.z
                ])
                goal_error = np.linalg.norm(final_ee_pos - self.goal_position)
                rospy.loginfo(f"📊 Final position accuracy:")
                rospy.loginfo(f"   Target: {self.goal_position}")
                rospy.loginfo(f"   Actual: {final_ee_pos}")
                rospy.loginfo(f"   Error: {goal_error*1000:.1f}mm")
                
                if goal_error > 0.01:  # 10mm tolerance
                    rospy.logwarn(f"⚠️ Large final position error: {goal_error*1000:.1f}mm")
                    rospy.loginfo("💡 Try increasing controller stiffness or allowing more settling time")
                else:
                    rospy.loginfo(f"✅ Excellent accuracy: {goal_error*1000:.1f}mm error")
            except Exception as e:
                rospy.logwarn(f"Could not verify final position: {e}")
            
            return True
                
        except Exception as e:
            rospy.logerr(f"Trajectory execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _visualize_trajectory(self, result):
        """Visualize the planned trajectory in RViz with enhanced details."""
        try:
            marker_array = MarkerArray()
            
            # Create trajectory line markers
            trajectory_marker = Marker()
            trajectory_marker.header.frame_id = "iiwa_link_0"
            trajectory_marker.header.stamp = rospy.Time.now()
            trajectory_marker.ns = "trajectory"
            trajectory_marker.id = 0
            trajectory_marker.type = Marker.LINE_STRIP
            trajectory_marker.action = Marker.ADD
            trajectory_marker.scale.x = 0.008  # Slightly thicker line
            trajectory_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)  # Green with transparency
            
            # Add trajectory points - recompute EE positions to match controller coordinate system
            for i in range(result.ee_positions.shape[1]):
                # For the first point, use the actual current position from controller
                if i == 0 and self.current_ee_pose is not None:
                    ee_pos = self.current_ee_pose
                else:
                    # Use the stored EE positions (may have coordinate system issues but shows trajectory shape)
                    ee_pos = result.ee_positions[:, i]
                
                point = Point()
                point.x = ee_pos[0]
                point.y = ee_pos[1] 
                point.z = ee_pos[2]
                trajectory_marker.points.append(point)
            
            marker_array.markers.append(trajectory_marker)
            
            # Add waypoint markers every 10th point for visual reference
            waypoint_skip = max(1, result.ee_positions.shape[1] // 15)  # Show ~15 waypoints max
            for i in range(0, result.ee_positions.shape[1], waypoint_skip):
                waypoint_marker = Marker()
                waypoint_marker.header.frame_id = "iiwa_link_0"
                waypoint_marker.header.stamp = rospy.Time.now()
                waypoint_marker.ns = "waypoints"
                waypoint_marker.id = 100 + i
                waypoint_marker.type = Marker.SPHERE
                waypoint_marker.action = Marker.ADD
                # Use correct position for waypoint markers
                if i == 0 and self.current_ee_pose is not None:
                    # First waypoint: use actual current position from controller
                    ee_pos = self.current_ee_pose
                else:
                    # Other waypoints: use stored positions (may have coordinate issues)
                    ee_pos = result.ee_positions[:, i]
                
                waypoint_marker.pose.position.x = ee_pos[0]
                waypoint_marker.pose.position.y = ee_pos[1]
                waypoint_marker.pose.position.z = ee_pos[2]
                waypoint_marker.pose.orientation.w = 1.0
                waypoint_marker.scale.x = waypoint_marker.scale.y = waypoint_marker.scale.z = 0.01
                # Color gradient from blue to green based on progress
                progress = i / (result.ee_positions.shape[1] - 1)
                waypoint_marker.color = ColorRGBA(r=0.0, g=progress, b=1.0-progress, a=0.6)
                marker_array.markers.append(waypoint_marker)
            
            # Create start point marker
            start_marker = Marker()
            start_marker.header.frame_id = "iiwa_link_0"
            start_marker.header.stamp = rospy.Time.now()
            start_marker.ns = "trajectory"
            start_marker.id = 1
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD
            start_marker.pose.position.x = result.ee_positions[0, 0]
            start_marker.pose.position.y = result.ee_positions[1, 0]
            start_marker.pose.position.z = result.ee_positions[2, 0]
            start_marker.pose.orientation.w = 1.0
            start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.02
            start_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue
            
            marker_array.markers.append(start_marker)
            
            # Create end point marker
            end_marker = Marker()
            end_marker.header.frame_id = "iiwa_link_0"
            end_marker.header.stamp = rospy.Time.now()
            end_marker.ns = "trajectory"
            end_marker.id = 2
            end_marker.type = Marker.SPHERE
            end_marker.action = Marker.ADD
            end_marker.pose.position.x = result.ee_positions[0, -1]
            end_marker.pose.position.y = result.ee_positions[1, -1]
            end_marker.pose.position.z = result.ee_positions[2, -1]
            end_marker.pose.orientation.w = 1.0
            end_marker.scale.x = end_marker.scale.y = end_marker.scale.z = 0.02
            end_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red
            
            marker_array.markers.append(end_marker)
            
            # Publish markers
            self.trajectory_marker_pub.publish(marker_array)
            
        except Exception as e:
            rospy.logwarn(f"Failed to visualize trajectory: {e}")
    
    def _visualize_goal(self):
        """Visualize the goal position in RViz."""
        try:
            goal_marker = Marker()
            goal_marker.header.frame_id = "iiwa_link_0"
            goal_marker.header.stamp = rospy.Time.now()
            goal_marker.ns = "goal"
            goal_marker.id = 0
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = self.goal_position[0]
            goal_marker.pose.position.y = self.goal_position[1]
            goal_marker.pose.position.z = self.goal_position[2]
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = self.goal_width * 2
            goal_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5)  # Yellow, semi-transparent
            
            self.goal_marker_pub.publish(goal_marker)
            
        except Exception as e:
            rospy.logwarn(f"Failed to visualize goal: {e}")
    
    def _clear_visualization(self):
        """Clear all trajectory and waypoint markers from RViz."""
        try:
            marker_array = MarkerArray()
            
            # Create delete markers for trajectory namespace
            for i in range(20):  # Clear up to 20 markers
                delete_marker = Marker()
                delete_marker.header.frame_id = "iiwa_link_0"
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.ns = "trajectory"
                delete_marker.id = i
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
            
            # Clear waypoint markers
            for i in range(200):  # Clear up to 200 waypoint markers
                delete_marker = Marker()
                delete_marker.header.frame_id = "iiwa_link_0"
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.ns = "waypoints"
                delete_marker.id = 100 + i
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
            
            self.trajectory_marker_pub.publish(marker_array)
            
        except Exception as e:
            rospy.logwarn(f"Failed to clear visualization: {e}")
    
    def set_goal_position(self, x: float, y: float, z: float):
        """Set a new goal position."""
        self.goal_position = np.array([x, y, z])
        rospy.loginfo(f"Goal position set to: {self.goal_position}")
        self._visualize_goal()
    
    def set_goal_width(self, width: float):
        """Set goal width (tolerance)."""
        self.goal_width = max(0.001, width)  # Minimum 1mm
        rospy.loginfo(f"Goal width set to: {self.goal_width}")
        self._visualize_goal()
    
    def print_status(self):
        """Print current status information."""
        print("\n" + "="*60)
        print("IIWA PLANNING INTERFACE STATUS")
        print("="*60)
        print(f"Mode: {'Simulation' if self.use_simulation else 'Real Robot'}")
        print(f"Controller: {self.controller_type}")
        print(f"Goal Position: {self.goal_position}")
        print(f"Goal Width: {self.goal_width}")
        
        if self.current_joint_state:
            joint_pos = self._extract_joint_positions(self.current_joint_state)
            if joint_pos is not None:
                print(f"Current Joint Positions: {np.round(joint_pos, 3)}")
        
        if self.current_ee_pose is not None:
            print(f"Current EE Position: {np.round(self.current_ee_pose, 3)}")
            distance_to_goal = np.linalg.norm(self.goal_position - self.current_ee_pose)
            print(f"Distance to Goal: {distance_to_goal:.3f}m")
        
        if self.last_planned_trajectory:
            print(f"Last Trajectory: {self.last_planned_trajectory.t.shape[0]} steps, "
                  f"{self.last_planned_trajectory.t[-1]:.2f}s duration")
            print(f"Trajectory Confirmed: {'Yes' if self.trajectory_confirmed else 'No'}")
        else:
            print("Last Trajectory: None")
        
        print("="*60)
    
    def move_to_safe_config(self):
        """Move robot to a safe configuration for planning."""
        rospy.loginfo("Moving robot to safe configuration...")
        
        # Create a simple trajectory to safe configuration
        current_state = self.get_current_robot_state()
        if current_state is None:
            rospy.logerr("Cannot get current state for safe move")
            return False
        
        rospy.loginfo(f"DEBUG: Current state shape: {current_state.shape}")
        rospy.loginfo(f"DEBUG: Current state: {current_state}")
        
        # Create initial state with safe joint configuration
        safe_state = np.concatenate([self.safe_joint_config, np.zeros(7)])
        rospy.loginfo(f"DEBUG: Safe state shape: {safe_state.shape}")
        rospy.loginfo(f"DEBUG: Safe state: {safe_state}")
        
        try:
            # Create a simple task to move to safe configuration
            from parameters.tasks.default_task_3d import TaskParams3D
            
            # Compute end-effector position for safe config
            if self.planner and self.planner.kinematics:
                # Use same FK as planner for consistency
                safe_ee_pos = self.planner.kinematics.fwd_kin(self.safe_joint_config)
                safe_ee_pos = np.array(safe_ee_pos).flatten()
                rospy.loginfo(f"DEBUG: Safe EE position: {safe_ee_pos}")
                
                # Debug: Check what TaskParams3D.create_reaching_task expects and returns
                rospy.loginfo("DEBUG: Creating reaching task...")
                # Split current_state into positions and velocities
                q_start = current_state[:7]   # First 7 elements are positions
                dq_start = current_state[7:]  # Last 7 elements are velocities
                rospy.loginfo(f"DEBUG: q_start shape: {q_start.shape}, dq_start shape: {dq_start.shape}")
                
                task = TaskParams3D.create_reaching_task(
                    q_start=q_start,
                    dq_start=dq_start,
                    goal_position=safe_ee_pos,
                    width=0.05  # Larger tolerance for safe move
                )
                
                rospy.loginfo(f"DEBUG: Task created. xi0 shape: {np.array(task.xi0).shape}")
                rospy.loginfo(f"DEBUG: Task xi0: {task.xi0}")
                rospy.loginfo(f"DEBUG: Task goal: {task.goal}")
                
                task_meta = {'width': 0.05, 'distance': 0.5, 'task_number': 0}
                
                rospy.loginfo("Planning safe configuration trajectory...")
                result = self.planner.solve(task, task_meta)
                
                # Store and visualize
                self.last_planned_trajectory = result
                self._visualize_trajectory(result)
                
                rospy.loginfo("Safe configuration trajectory planned. Press 'c' to confirm and 'e' to execute.")
                return True
                
        except Exception as e:
            rospy.logerr(f"Failed to plan safe configuration move: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_help(self):
        """Print available commands."""
        print("\n" + "="*60)
        print("AVAILABLE COMMANDS")
        print("="*60)
        print("p  - Plan new trajectory from current state to goal")
        print("c  - Confirm current trajectory (shows details & visualization)")
        print("e  - Execute confirmed trajectory")
        print("m  - Move to safe configuration (recommended if robot at zero)")
        print("s  - Show current status")
        print("g  - Set goal position (interactive)")
        print("w  - Set goal width (interactive)")
        print("v  - Clear trajectory visualization")
        print("h  - Show this help")
        print("q  - Quit")
        print("="*60)
    
    def run_keyboard_interface(self):
        """Run the interactive keyboard interface."""
        print("\n🤖 IIWA Planning Interface Started!")
        print("Press 'h' for help, 'q' to quit")
        
        try:
            # Set terminal to raw mode for immediate key capture
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._get_raw_terminal_settings())
            
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'q':
                        print("\nQuitting...")
                        break
                    elif key == 'h':
                        self.print_help()
                    elif key == 's':
                        self.print_status()
                    elif key == 'p':
                        print("\n🎯 Planning trajectory...")
                        success = self.plan_trajectory()
                        if success:
                            print("✅ Trajectory planned! Press 'c' to confirm, then 'e' to execute.")
                        else:
                            print("❌ Trajectory planning failed!")
                    elif key == 'c':
                        if self.last_planned_trajectory:
                            self.trajectory_confirmed = True
                            print("\n✅ Trajectory confirmed!")
                            
                            # Show trajectory details for confirmation
                            traj = self.last_planned_trajectory
                            print(f"📊 Trajectory Details:")
                            print(f"  - Duration: {traj.t[-1]:.3f}s")
                            print(f"  - Waypoints: {traj.q.shape[1]}")
                            print(f"  - Start position: {traj.q[:, 0]}")
                            print(f"  - Final position: {traj.q[:, -1]}")
                            
                            # Show end-effector trajectory using actual positions
                            if hasattr(traj, 'ee_positions'):
                                # Use actual current EE position instead of computed one
                                start_ee = self.current_ee_pose if self.current_ee_pose is not None else traj.ee_positions[:, 0]
                                final_ee = traj.ee_positions[:, -1]
                                distance = np.linalg.norm(final_ee - start_ee)
                                print(f"  - EE start: [{start_ee[0]:.3f}, {start_ee[1]:.3f}, {start_ee[2]:.3f}] (from controller)")
                                print(f"  - EE final: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}] (computed)")
                                print(f"  - EE distance: {distance:.3f}m")
                            
                            # Visualize trajectory in RViz
                            self._visualize_trajectory(traj)
                            self._visualize_goal()
                            print("🎯 Trajectory visualized in RViz!")
                            print("Press 'e' to execute.")
                        else:
                            print("\n❌ No trajectory to confirm. Plan one first with 'p'.")
                    elif key == 'e':
                        print("\n🚀 Executing trajectory...")
                        success = self.execute_trajectory()
                        if success:
                            print("✅ Trajectory executed successfully!")
                        else:
                            print("❌ Trajectory execution failed!")
                    elif key == 'g':
                        self._interactive_set_goal()
                    elif key == 'w':
                        self._interactive_set_width()
                    elif key == 'v':
                        print("\n🧹 Clearing trajectory visualization...")
                        self._clear_visualization()
                        print("✅ Visualization cleared!")
                    elif key == 'm':
                        print("\n🔧 Moving to safe configuration...")
                        success = self.move_to_safe_config()
                        if success:
                            print("✅ Safe configuration trajectory planned! Press 'c' to confirm, then 'e' to execute.")
                        else:
                            print("❌ Failed to plan safe configuration move!")
                    elif key in ['\r', '\n']:
                        continue  # Ignore enter key
                    else:
                        print(f"\nUnknown command: '{key}'. Press 'h' for help.")
                
                # Small sleep to prevent high CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def _get_raw_terminal_settings(self):
        """Get terminal settings for raw input."""
        new_settings = termios.tcgetattr(sys.stdin)
        new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)
        return new_settings
    
    def _interactive_set_goal(self):
        """Interactive goal position setting."""
        # Temporarily restore normal terminal mode for input
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        
        try:
            print(f"\nCurrent goal: {self.goal_position}")
            x_str = input("Enter X coordinate (m): ")
            y_str = input("Enter Y coordinate (m): ")
            z_str = input("Enter Z coordinate (m): ")
            
            x = float(x_str)
            y = float(y_str)
            z = float(z_str)
            
            self.set_goal_position(x, y, z)
            print("Goal position updated!")
            
        except ValueError:
            print("Invalid input. Goal position unchanged.")
        except KeyboardInterrupt:
            print("\nGoal setting cancelled.")
        finally:
            # Return to raw mode
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._get_raw_terminal_settings())
    
    def _interactive_set_width(self):
        """Interactive goal width setting."""
        # Temporarily restore normal terminal mode for input
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        
        try:
            print(f"\nCurrent goal width: {self.goal_width}")
            width_str = input("Enter goal width (m): ")
            
            width = float(width_str)
            self.set_goal_width(width)
            print("Goal width updated!")
            
        except ValueError:
            print("Invalid input. Goal width unchanged.")
        except KeyboardInterrupt:
            print("\nWidth setting cancelled.")
        finally:
            # Return to raw mode
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._get_raw_terminal_settings())


def main():
    """Main function."""
    try:
        # Create and run the interface
        interface = IIWAPlanningInterface()
        
        # Start the keyboard interface in the main thread
        interface.run_keyboard_interface()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

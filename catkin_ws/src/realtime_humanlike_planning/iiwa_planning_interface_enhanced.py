#!/usr/bin/env python3
"""
Enhanced IIWA Planning Interface with Z-Plane Targeting
Provides streamlined interface with two planning modes:
1. Goal planning mode - plan to specific 3D position  
2. Z-plane planning mode - plan to target z-height plane
"""

import rospy
import numpy as np
import time
import yaml
import os
import sys
import termios
import select
from typing import Optional
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float64MultiArray
from tf2_ros import TransformListener, Buffer
import actionlib
from iiwa_impedance_control.msg import CartesianTrajectoryExecutionAction, CartesianTrajectoryExecutionGoal

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
from enhanced_controller_manager import EnhancedControllerManager


class EnhancedIIWAPlanningInterface:
    """Enhanced planning interface with dual planning modes and visualization."""
    
    def _safe_input(self, prompt):
        """Safe input function that temporarily restores terminal echo."""
        try:
            # Get current terminal settings
            current_settings = termios.tcgetattr(sys.stdin)
            # Restore canonical and echo mode temporarily
            normal_settings = current_settings.copy()
            normal_settings[3] |= (termios.ICANON | termios.ECHO)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, normal_settings)
            
            # Get input with echo enabled
            result = input(prompt)
            
            # Restore raw mode
            raw_settings = current_settings.copy()
            raw_settings[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, raw_settings)
            
            return result
        except Exception:
            # Fallback to regular input
            return input(prompt)
    
    def __init__(self):
        """Initialize the enhanced planning interface."""
        rospy.init_node('iiwa_planning_interface_enhanced', anonymous=True)
        
        # Configuration
        self.robot_name = rospy.get_param('~robot_name', 'iiwa')
        self.use_simulation = rospy.get_param('~use_simulation', True)
        
        # Load configuration from YAML file
        self._load_configuration()
        
        # Planning modes: 'goal' or 'z_plane'
        self.planning_mode = 'goal'  # Start with goal planning
        
        # State variables
        self.current_joint_state = None
        self.current_ee_pose = None
        self.last_planned_trajectory = None
        self.trajectory_confirmed = False
        
        # Initialize components
        self._init_components()
        
        rospy.loginfo("Enhanced IIWA Planning Interface initialized!")
        rospy.loginfo(f"Environment: {'Simulation' if self.use_simulation else 'Real Robot'}")
        rospy.loginfo(f"Planning mode: {self.planning_mode}")
        
        # Apply initial settings
        self._apply_initial_settings()
        
        # Initial visualization
        self._visualize_current_mode()
        
    def _load_configuration(self):
        """Load configuration from YAML file."""
        config_path = os.path.join(current_dir, 'config', 'planning_config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load configuration values
            self.initial_joint_state = np.array(config['initial_joint_state'])
            self.initial_stiffness = config['initial_cartesian_stiffness']
            self.initial_damping = config['initial_cartesian_damping']
            self.goal_width = config['initial_target_width']
            self.goal_position = np.array(config['default_goal_position'])
            self.target_z_plane = config['default_z_plane_height']
            
            # Planning parameters
            planning_config = config['planning']
            self.horizon = planning_config['horizon']
            self.solver_type = planning_config['solver_type']
            self.use_fitts_law = planning_config.get('use_fitts_law', True)
            
            # Visualization settings
            viz_config = config['visualization']
            self.z_plane_size = viz_config['z_plane_size']
            self.z_plane_alpha = viz_config['z_plane_alpha']
            self.target_scale = viz_config['target_scale']
            self.show_initial_trajectory = viz_config['show_initial_trajectory']
            self.initial_trajectory_alpha = viz_config['initial_trajectory_alpha']
            
            rospy.loginfo("✅ Configuration loaded from YAML file")
            
        except Exception as e:
            rospy.logwarn(f"Failed to load configuration: {e}")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self.initial_joint_state = np.array([0.0, 0.280260657, 0.0, -1.53136477, 0.0, 1.31590271, 0.0])
        self.initial_stiffness = [0.1, 0.1, 300, 300, 300, 300]
        self.initial_damping = [2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
        self.goal_width = 0.015
        self.goal_position = np.array([0.5, 0.2, 0.6])
        self.target_z_plane = 0.6
        self.z_plane_size = 1.0
        self.z_plane_alpha = 0.3
        self.target_scale = 0.05
        self.horizon = 30
        self.solver_type = "mumps"
        self.use_fitts_law = True
        self.show_initial_trajectory = True
        self.initial_trajectory_alpha = 0.6
        rospy.loginfo("Using default configuration values")
    
    def _init_components(self):
        """Initialize ROS components."""
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        
        # Publishers - match RViz configuration topics
        self.trajectory_pub = rospy.Publisher('/iiwa_planning_interface/trajectory_markers', MarkerArray, queue_size=10)
        self.goal_pub = rospy.Publisher('/iiwa_planning_interface/goal_marker', Marker, queue_size=10)
        self.z_plane_pub = rospy.Publisher('/iiwa_planning_interface/z_plane_marker', Marker, queue_size=10)
        self.target_pub = rospy.Publisher('/iiwa_planning_interface/target_marker', Marker, queue_size=10)
        self.initial_traj_pub = rospy.Publisher('/iiwa_planning_interface/initial_trajectory_markers', MarkerArray, queue_size=10)
        
        # Subscribers
        self.joint_state_sub = rospy.Subscriber(f'/{self.robot_name}/joint_states', JointState, self._joint_state_callback)
        
        # Action client
        self.cartesian_action_client = actionlib.SimpleActionClient(
            f'/{self.robot_name}/CartesianImpedanceController/cartesian_trajectory_execution_action',
            CartesianTrajectoryExecutionAction
        )
        
        # Wait for action server
        rospy.loginfo("Waiting for Cartesian trajectory action server...")
        if self.cartesian_action_client.wait_for_server(timeout=rospy.Duration(10)):
            rospy.loginfo("✅ Action server connected")
        else:
            rospy.logwarn("❌ Action server not available. Trajectories may not execute.")
        
        # Enhanced controller manager
        self.controller_manager = EnhancedControllerManager(self.robot_name, self.use_simulation)
        
        # Initialize planner
        try:
            rospy.loginfo("🔧 Initializing planner with horizon=%d, solver=%s", self.horizon, self.solver_type)
            
            reward_params = VanHallRewardParams3D.default()
            reward_params.use_fitts_law = self.use_fitts_law
            
            rospy.loginfo("🔧 Creating planner instance...")
            self.planner = VanHallHumanReaching3D_Optimized(
                H=self.horizon,
                reward_params=reward_params,
                solver_type=self.solver_type
            )
            
            rospy.loginfo(f"✅ Planner initialized successfully (Fitts Law: {self.use_fitts_law})")
        except ImportError as e:
            rospy.logerr(f"Import error while initializing planner: {e}")
            rospy.logerr("Check if planning modules are in Python path")
            self.planner = None
        except Exception as e:
            rospy.logerr(f"Failed to initialize planner: {e}")
            import traceback
            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            self.planner = None
    
    def _apply_initial_settings(self):
        """Apply initial settings from configuration."""
        # Switch to impedance control mode
        initial_mode = "impedance_control" if self.use_simulation else "hand_guiding"
        if self.controller_manager.switch_to_mode(initial_mode):
            rospy.loginfo(f"✅ Started in {initial_mode} mode")
        else:
            rospy.logwarn(f"❌ Failed to start in {initial_mode} mode")
    
    def _joint_state_callback(self, msg: JointState):
        """Callback for joint state updates."""
        # Extract joint positions in proper order - critical for planner!
        expected_joints = [
            'iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4',
            'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7'
        ]
        
        try:
            joint_positions = np.zeros(7)
            joint_velocities = np.zeros(7)
            
            # Extract positions in correct order
            for i, joint_name in enumerate(expected_joints):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    joint_positions[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        joint_velocities[i] = msg.velocity[idx]
                else:
                    rospy.logwarn_once(f"Joint {joint_name} not found in joint state message")
                    return  # Don't update if we can't get all joints
            
            # Store as [q1...q7, dq1...dq7] for planner
            self.current_joint_state = np.concatenate([joint_positions, joint_velocities])
            
            # Debug first few callbacks
            if not hasattr(self, '_joint_callback_count'):
                self._joint_callback_count = 0
                
            self._joint_callback_count += 1
            if self._joint_callback_count <= 3:
                rospy.loginfo(f"🔄 DEBUG: Joint state callback #{self._joint_callback_count}")
                rospy.loginfo(f"📊 DEBUG: Received joints: {msg.name}")
                rospy.loginfo(f"📊 DEBUG: Extracted positions: {joint_positions}")
                rospy.loginfo(f"📊 DEBUG: Final joint state shape: {self.current_joint_state.shape}")
            
            # Update end-effector pose
            self._update_current_ee_pose()
            
        except Exception as e:
            rospy.logwarn(f"Failed to process joint state: {e}")
            # Don't update joint state if processing failed
    
    def _update_current_ee_pose(self):
        """Update current end-effector pose using TF."""
        try:
            transform = self.tf_buffer.lookup_transform('iiwa_link_0', 'iiwa_link_ee', rospy.Time(0), rospy.Duration(1.0))
            pos = transform.transform.translation
            self.current_ee_pose = np.array([pos.x, pos.y, pos.z])
        except Exception as e:
            if self.current_ee_pose is None:
                rospy.logwarn_once(f"Failed to get EE pose: {e}")
    
    def switch_planning_mode(self, mode: str):
        """Switch between planning modes."""
        if mode not in ['goal', 'z_plane']:
            rospy.logwarn(f"Invalid planning mode: {mode}")
            return
            
        self.planning_mode = mode
        
        # Reset planner horizon to default when switching modes
        if self.planner is not None and hasattr(self.planner, 'H'):
            original_horizon = 30  # Default horizon
            if self.planner.H != original_horizon:
                rospy.loginfo(f"🔄 Resetting planner horizon from {self.planner.H} to {original_horizon} for new planning mode")
                self.planner.H = original_horizon
                
                # Update bounds computer with default horizon
                self.planner.bounds_computer = self.planner.bounds_computer.__class__(
                    self.planner.H, self.planner.nq, self.planner.NST, self.planner.NGOAL, self.planner.cov_min,
                    np.array(self.planner.model.joint_limits_lower),
                    np.array(self.planner.model.joint_limits_upper),
                    np.array(self.planner.model.joint_velocity_limits),
                    np.array(self.planner.model.joint_effort_limits)
                )
                
                # Rebuild solver template with default horizon
                self.planner._build_solver_template()
                rospy.loginfo(f"✅ Planner reset for {mode} planning mode")
        
        self._clear_all_visualizations()
        self._visualize_current_mode()
        
        rospy.loginfo(f"🔄 Switched to {mode} planning mode")
        print(f"\n🔄 Switched to {mode} planning mode")
    
    def _visualize_current_mode(self):
        """Visualize current mode (goal or z-plane)."""
        if self.planning_mode == 'goal':
            self._visualize_goal()
        elif self.planning_mode == 'z_plane':
            self._visualize_z_plane()
    
    def plan_trajectory(self) -> bool:
        """Plan trajectory based on current mode."""
        rospy.loginfo("🚀 Starting trajectory planning...")
        rospy.loginfo(f"📊 Planning mode: {self.planning_mode}")
        
        # Comprehensive pre-planning checks
        rospy.loginfo(f"🔍 Planning debug: joint_state={self.current_joint_state is not None}, planner={self.planner is not None}")
        
        if self.current_joint_state is None:
            rospy.logerr("❌ Cannot plan: missing joint state - robot may not be publishing joint states")
            rospy.logerr("💡 Check if /iiwa/joint_states topic is publishing")
            return False
        
        if self.planner is None:
            rospy.logerr("❌ Cannot plan: planner initialization failed - check planner import and dependencies")
            return False
        
        # Debug current joint state format
        rospy.loginfo(f"📊 DEBUG: joint_state type: {type(self.current_joint_state)}")
        if hasattr(self.current_joint_state, 'shape'):
            rospy.loginfo(f"📊 DEBUG: joint_state shape: {self.current_joint_state.shape}")
        elif hasattr(self.current_joint_state, '__len__'):
            rospy.loginfo(f"📊 DEBUG: joint_state length: {len(self.current_joint_state)}")
        
        # Debug planner state
        if hasattr(self.planner, 'H'):
            rospy.loginfo(f"📊 DEBUG: Planner horizon: {self.planner.H}")
        if hasattr(self.planner, 'rp') and hasattr(self.planner.rp, 'use_fitts_law'):
            rospy.loginfo(f"📊 DEBUG: Fitts law enabled: {self.planner.rp.use_fitts_law}")
        
        # Execute planning based on mode
        try:
            rospy.loginfo(f"📋 Executing {self.planning_mode} planning...")
            
            if self.planning_mode == 'goal':
                success = self._plan_to_goal()
            elif self.planning_mode == 'z_plane':
                success = self._plan_to_z_plane()
            else:
                rospy.logerr(f"❌ Unknown planning mode: {self.planning_mode}")
                return False
            
            if success:
                rospy.loginfo("🎉 Planning completed successfully!")
            else:
                rospy.logerr("💥 Planning failed!")
                
            return success
            
        except Exception as e:
            rospy.logerr(f"❌ Planning failed with exception: {e}")
            import traceback
            rospy.logerr(f"📋 Full traceback:\n{traceback.format_exc()}")
            return False
    
    def _plan_to_goal(self) -> bool:
        """Plan trajectory to goal position."""
        # Ensure goal_position is a proper 3D numpy array
        goal_pos = np.asarray(self.goal_position, dtype=float).flatten()
        
        if goal_pos.size != 3:
            rospy.logerr(f"Invalid goal position size: {goal_pos.size}, expected 3. Resetting to default.")
            goal_pos = np.array([0.5, 0.2, 0.6], dtype=float)
            self.goal_position = goal_pos  # Update the stored goal position
        
        # Get current robot state with extensive debugging
        rospy.loginfo("🔍 DEBUG: Getting current robot state...")
        
        if self.current_joint_state is None:
            rospy.logerr("❌ No current joint state available from ROS topic")
            return False
        
        # Extract joint positions and velocities
        joint_positions = self.current_joint_state[:7] if len(self.current_joint_state) >= 7 else None
        joint_velocities = self.current_joint_state[7:14] if len(self.current_joint_state) >= 14 else np.zeros(7)
        
        if joint_positions is None:
            rospy.logerr("❌ Cannot extract joint positions from current joint state")
            return False
        
        # Comprehensive debugging
        rospy.loginfo(f"📊 DEBUG: Current joint state shape: {self.current_joint_state.shape}")
        rospy.loginfo(f"📊 DEBUG: Joint positions: {joint_positions}")
        rospy.loginfo(f"📊 DEBUG: Joint velocities: {joint_velocities}")
        rospy.loginfo(f"🎯 DEBUG: Goal position: {goal_pos}")
        rospy.loginfo(f"📏 DEBUG: Goal width: {self.goal_width}")
        
        # Check for singular configurations
        if np.allclose(joint_positions, 0.0, atol=0.1):
            rospy.logwarn("⚠️ Robot is near zero configuration, which may cause kinematics issues")
        
        # Update current EE pose for distance calculation
        self._update_current_ee_pose()
        
        # Calculate distance for debugging and task metadata
        if self.current_ee_pose is not None:
            distance = np.linalg.norm(goal_pos - self.current_ee_pose)
            rospy.loginfo(f"📏 DEBUG: Current EE pose: {self.current_ee_pose}")
            rospy.loginfo(f"📏 DEBUG: Distance to goal: {distance:.3f}m")
        else:
            distance = 0.5  # Default assumption
            rospy.logwarn("⚠️ Current EE pose not available, using default distance")
        
        # Transform goal position for planner if needed
        goal_planner_frame = goal_pos
        if hasattr(self.planner, 'kinematics') and hasattr(self.planner.kinematics, 'transform_from_controller_frame'):
            try:
                goal_planner_frame = self.planner.kinematics.transform_from_controller_frame(goal_pos)
                rospy.loginfo(f"🔄 DEBUG: Goal transformed from {goal_pos} to {goal_planner_frame}")
            except Exception as e:
                rospy.logwarn(f"⚠️ Could not transform goal position: {e}")
        
        # Create task parameters
        rospy.loginfo("🔧 DEBUG: Creating task parameters...")
        task_params = TaskParams3D.create_reaching_task(
            q_start=joint_positions,
            dq_start=joint_velocities, 
            goal_position=goal_planner_frame,
            width=self.goal_width
        )
        
        rospy.loginfo(f"✅ DEBUG: Task created - xi0 shape: {task_params.xi0.shape}")
        rospy.loginfo(f"📊 DEBUG: Task xi0 (positions): {task_params.xi0[:7]}")
        rospy.loginfo(f"📊 DEBUG: Task xi0 (velocities): {task_params.xi0[7:]}")
        rospy.loginfo(f"🎯 DEBUG: Task goal: {task_params.goal}")
        rospy.loginfo(f"📏 DEBUG: Task width: {task_params.width}")
        
        # Verify starting EE position that planner will compute
        if hasattr(self.planner, 'kinematics') and self.planner.kinematics:
            try:
                planner_start_ee_internal = self.planner.kinematics.fwd_kin(task_params.xi0[:7])
                planner_start_ee_internal = np.array(planner_start_ee_internal).flatten()
                
                # Transform to controller frame for comparison if method exists
                if hasattr(self.planner.kinematics, 'transform_to_controller_frame'):
                    planner_start_ee = self.planner.kinematics.transform_to_controller_frame(planner_start_ee_internal)
                else:
                    planner_start_ee = planner_start_ee_internal
                
                rospy.loginfo(f"🤖 DEBUG: Planner will compute start EE as: {planner_start_ee}")
                
                if self.current_ee_pose is not None:
                    error = np.linalg.norm(planner_start_ee - self.current_ee_pose)
                    rospy.loginfo(f"📊 DEBUG: EE position error: {error*1000:.1f}mm")
                    if error > 0.05:  # 5cm threshold
                        rospy.logwarn(f"⚠️ Large EE position discrepancy: {error*1000:.1f}mm")
                        
            except Exception as e:
                rospy.logwarn(f"⚠️ Could not compute planner start EE: {e}")
        
        # Task metadata for better planning
        task_meta = {
            'width': self.goal_width,
            'distance': distance,
            'task_number': 1
        }
        
        rospy.loginfo(f"📦 DEBUG: Task metadata: {task_meta}")
        
        # Solve with timing and detailed error handling
        rospy.loginfo("🚀 DEBUG: Starting trajectory optimization...")
        
        # Clear initial trajectory when starting optimization so user can see the difference
        self._clear_initial_trajectory()
        rospy.sleep(0.1)  # Brief pause for visual clarity
        
        try:
            import time
            start_time = time.time()
            
            result = self.planner.solve(task_params, task_meta, start_ee_pos=self.current_ee_pose)
            
            solve_time = time.time() - start_time
            rospy.loginfo(f"⏱️ DEBUG: Solve time: {solve_time:.3f}s")
            
            # Check solver statistics
            if hasattr(result, 'solver_stats'):
                rospy.loginfo(f"📊 DEBUG: Solver stats: {result.solver_stats}")
            
            success = getattr(result, 'success', True)
            rospy.loginfo(f"📊 DEBUG: Solver success: {success}")
            
            if success:
                rospy.loginfo(f"✅ DEBUG: Trajectory has {result.q.shape[1]} waypoints over {result.t[-1]:.3f}s")
                if hasattr(result, 'ee_positions'):
                    final_ee = result.ee_positions[:, -1]
                    final_error = np.linalg.norm(final_ee - goal_pos)
                    rospy.loginfo(f"🎯 DEBUG: Final EE position: {final_ee}")
                    rospy.loginfo(f"📏 DEBUG: Final goal error: {final_error*1000:.2f}mm")
            
            # Store result if successful
            if success:
                self.last_planned_trajectory = result
                self.trajectory_confirmed = False
                rospy.loginfo("✅ Trajectory to goal planned successfully!")
                
                # Visualize the planned trajectory immediately
                self._visualize_trajectory(result)
                self._visualize_goal()
            else:
                rospy.logerr("❌ Failed to plan trajectory to goal - solver did not converge")
                
            return success
            
        except Exception as e:
            rospy.logerr(f"❌ DEBUG: Solver exception: {e}")
            import traceback
            rospy.logerr(f"📋 DEBUG: Traceback:\n{traceback.format_exc()}")
            return False
    
    def _plan_to_z_plane(self) -> bool:
        """Plan trajectory to z-plane."""
        if self.current_ee_pose is None:
            rospy.logerr("Current EE pose not available")
            return False
        
        # Get current robot state with same validation as goal planning
        rospy.loginfo("🔍 DEBUG: Getting current robot state for z-plane...")
        
        if self.current_joint_state is None:
            rospy.logerr("❌ No current joint state available from ROS topic")
            return False
        
        # Extract joint positions and velocities (same as goal planning)
        joint_positions = self.current_joint_state[:7] if len(self.current_joint_state) >= 7 else None
        joint_velocities = self.current_joint_state[7:14] if len(self.current_joint_state) >= 14 else np.zeros(7)
        
        if joint_positions is None:
            rospy.logerr("❌ Cannot extract joint positions from current joint state")
            return False
        
        # Project current position onto target z-plane
        target_position = self.current_ee_pose.copy()
        target_position[2] = self.target_z_plane
        
        # Defensive fix: ensure target_position is a proper 3D numpy array
        target_pos = np.asarray(target_position, dtype=float)
        if target_pos.shape != (3,):
            rospy.logerr(f"Invalid target position shape: {target_pos.shape}, expected (3,). Using fallback.")
            target_pos = np.array([0.5, 0.0, self.target_z_plane], dtype=float)  # Safe fallback
        
        rospy.loginfo(f"🎯 DEBUG: Z-plane target: {target_pos}")
        rospy.loginfo(f"📏 DEBUG: Current EE: {self.current_ee_pose}")
        
        task_params = TaskParams3D.create_reaching_task(
            q_start=joint_positions,
            dq_start=joint_velocities,
            goal_position=target_pos,
            width=self.goal_width
        )
        
        rospy.loginfo(f"✅ DEBUG: Z-plane task created - xi0 shape: {task_params.xi0.shape}")
        
        # Add task metadata for better planning
        distance = np.linalg.norm(target_pos - self.current_ee_pose)
        task_meta = {
            'width': self.goal_width,
            'distance': distance
        }
        
        rospy.loginfo(f"📦 DEBUG: Z-plane task metadata: {task_meta}")
        
        result = self.planner.solve(task_params, task_meta)
        
        if result.success:
            self.last_planned_trajectory = result
            self.trajectory_confirmed = False
            rospy.loginfo("✅ Trajectory to z-plane planned")
            
            # Visualize the planned trajectory immediately
            self._visualize_trajectory(result)
            self._visualize_z_plane()
            return True
        else:
            rospy.logerr("❌ Failed to plan trajectory to z-plane")
            return False
    
    def execute_trajectory(self) -> bool:
        """Execute the planned trajectory by publishing reference poses at control rate."""
        if not self.trajectory_confirmed or self.last_planned_trajectory is None:
            rospy.logerr("No confirmed trajectory to execute")
            return False
        
        # Ensure we're in impedance control mode for execution
        if self.controller_manager.get_current_mode() != "impedance_control":
            rospy.loginfo("🔄 Switching to impedance control for trajectory execution")
            if not self.controller_manager.switch_to_mode("impedance_control"):
                rospy.logerr("❌ Failed to switch to impedance control")
                return False
        
        rospy.loginfo("🚀 Executing human-like trajectory with real-time reference poses...")
        
        try:
            trajectory = self.last_planned_trajectory
            
            # Create publisher for reference poses
            from geometry_msgs.msg import PoseStamped
            ref_pose_pub = rospy.Publisher(
                '/iiwa/CartesianImpedanceController/reference_pose', 
                PoseStamped, 
                queue_size=10
            )
            
            # Wait for publisher to be ready
            rospy.sleep(0.5)
            
            # Execute trajectory by publishing reference poses at control rate
            rate = rospy.Rate(100)  # 100 Hz for smooth impedance control
            start_time = rospy.Time.now()
            
            rospy.loginfo(f"📡 Publishing human-like trajectory: {trajectory.q.shape[1]} waypoints over {trajectory.t[-1]:.3f}s at 100Hz")
            
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
                        
                        # Transform to controller frame if method exists
                        if hasattr(self.planner.kinematics, 'transform_to_controller_frame'):
                            ee_pos = self.planner.kinematics.transform_to_controller_frame(ee_pos_internal)
                        else:
                            ee_pos = ee_pos_internal
                        
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
                        
                        # Publish reference pose at control rate
                        ref_pose_pub.publish(ref_pose)
                        
                        if i % 50 == 0:  # Log every 50th waypoint
                            rospy.loginfo(f"📡 Publishing waypoint {i+1}/{trajectory.q.shape[1]} at t={trajectory.t[i]:.3f}s")
                        
                    except Exception as e:
                        rospy.logwarn(f"Failed to compute FK for waypoint {i}: {e}")
                
                rate.sleep()
            
            rospy.loginfo("✅ Complete human-like trajectory published successfully!")
            
            # Wait for the robot to settle at the final position
            rospy.loginfo("⏳ Waiting 2.0s for robot to settle at final position...")
            rospy.sleep(2.0)
            
            # Switch back to appropriate mode after execution
            if not self.use_simulation:
                rospy.loginfo("🔄 Switching back to hand-guiding mode")
                self.controller_manager.switch_to_mode("hand_guiding")
            
            return True
                
        except Exception as e:
            rospy.logerr(f"Trajectory execution failed: {e}")
            import traceback
            rospy.logerr(f"📋 Execution traceback:\n{traceback.format_exc()}")
            return False
    
    def update_impedance_stiffness(self):
        """Update impedance stiffness parameters interactively."""
        try:
            print("\n🔧 Current Cartesian Stiffness Parameters:")
            labels = ['X', 'Y', 'Z', 'A', 'B', 'C']
            units = ['N/m', 'N/m', 'N/m', 'Nm/rad', 'Nm/rad', 'Nm/rad']
            
            for label, value, unit in zip(labels, self.initial_stiffness, units):
                print(f"  {label}: {value} {unit}")
            
            print("\nEnter new values (press Enter to keep current):")
            new_stiffness = []
            
            # Temporarily restore normal terminal mode for visible input
            if hasattr(self, 'old_settings'):
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
            try:
                for label, current_val, unit in zip(labels, self.initial_stiffness, units):
                    while True:
                        try:
                            user_input = input(f"{label} [{current_val} {unit}]: ").strip()
                            if user_input == "":
                                new_stiffness.append(current_val)
                                break
                            else:
                                value = float(user_input)
                                new_stiffness.append(value)
                                break
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                
                # Update via controller manager
                if self.controller_manager.update_stiffness_parameters(new_stiffness, self.initial_damping):
                    self.initial_stiffness = new_stiffness
                    print("✅ Impedance parameters updated!")
                else:
                    print("❌ Failed to update impedance parameters")
                    
            finally:
                # Restore raw terminal mode for main interface
                if hasattr(self, 'old_settings'):
                    tty.setraw(sys.stdin.fileno())
                
        except Exception as e:
            rospy.logerr(f"Failed to update parameters: {e}")
    
    def _visualize_trajectory(self, trajectory):
        """Visualize planned trajectory with enhanced markers."""
        try:
            marker_array = MarkerArray()
            
            # Main trajectory line
            line_marker = Marker()
            line_marker.header.frame_id = "iiwa_link_0"
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "planned_trajectory"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.010  # Slightly thicker line
            line_marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.9)  # Bright green
            
            if hasattr(trajectory, 'ee_positions') and trajectory.ee_positions.size > 0:
                rospy.loginfo(f"📊 Visualizing trajectory with {trajectory.ee_positions.shape[1]} waypoints")
                
                for i in range(trajectory.ee_positions.shape[1]):
                    point = Point()
                    point.x = trajectory.ee_positions[0, i]
                    point.y = trajectory.ee_positions[1, i] 
                    point.z = trajectory.ee_positions[2, i]
                    line_marker.points.append(point)
                
                # Add waypoint markers every 10th point
                waypoint_skip = max(1, trajectory.ee_positions.shape[1] // 15)  # Show ~15 waypoints
                for i in range(0, trajectory.ee_positions.shape[1], waypoint_skip):
                    waypoint_marker = Marker()
                    waypoint_marker.header.frame_id = "iiwa_link_0"
                    waypoint_marker.header.stamp = rospy.Time.now()
                    waypoint_marker.ns = "trajectory_waypoints"
                    waypoint_marker.id = i
                    waypoint_marker.type = Marker.SPHERE
                    waypoint_marker.action = Marker.ADD
                    
                    waypoint_marker.pose.position.x = trajectory.ee_positions[0, i]
                    waypoint_marker.pose.position.y = trajectory.ee_positions[1, i]
                    waypoint_marker.pose.position.z = trajectory.ee_positions[2, i]
                    waypoint_marker.pose.orientation.w = 1.0
                    
                    waypoint_marker.scale.x = waypoint_marker.scale.y = waypoint_marker.scale.z = 0.015
                    
                    # Color gradient from blue (start) to green (end)
                    progress = i / (trajectory.ee_positions.shape[1] - 1)
                    waypoint_marker.color = ColorRGBA(0.0, progress, 1.0-progress, 0.8)
                    marker_array.markers.append(waypoint_marker)
                
                # Start point marker (blue)
                start_marker = Marker()
                start_marker.header.frame_id = "iiwa_link_0"
                start_marker.header.stamp = rospy.Time.now()
                start_marker.ns = "trajectory_points"
                start_marker.id = 1
                start_marker.type = Marker.SPHERE
                start_marker.action = Marker.ADD
                
                start_marker.pose.position.x = trajectory.ee_positions[0, 0]
                start_marker.pose.position.y = trajectory.ee_positions[1, 0]
                start_marker.pose.position.z = trajectory.ee_positions[2, 0]
                start_marker.pose.orientation.w = 1.0
                
                start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.025
                start_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue
                marker_array.markers.append(start_marker)
                
                # End point marker (red)
                end_marker = Marker()
                end_marker.header.frame_id = "iiwa_link_0"
                end_marker.header.stamp = rospy.Time.now()
                end_marker.ns = "trajectory_points"
                end_marker.id = 2
                end_marker.type = Marker.SPHERE
                end_marker.action = Marker.ADD
                
                end_marker.pose.position.x = trajectory.ee_positions[0, -1]
                end_marker.pose.position.y = trajectory.ee_positions[1, -1]
                end_marker.pose.position.z = trajectory.ee_positions[2, -1]
                end_marker.pose.orientation.w = 1.0
                
                end_marker.scale.x = end_marker.scale.y = end_marker.scale.z = 0.025
                end_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red
                marker_array.markers.append(end_marker)
                
            else:
                rospy.logwarn("Trajectory has no ee_positions to visualize")
            
            marker_array.markers.append(line_marker)
            self.trajectory_pub.publish(marker_array)
            rospy.loginfo("📊 Trajectory visualization published to RViz")
            
        except Exception as e:
            rospy.logwarn(f"Trajectory visualization failed: {e}")
            import traceback
            rospy.logwarn(f"Visualization traceback: {traceback.format_exc()}")
    
    def _visualize_initial_trajectory(self):
        """Visualize initial trajectory guess in orange."""
        if not self.show_initial_trajectory:
            return
            
        try:
            # Create a simple initial trajectory for visualization
            # This shows the straight-line interpolation before optimization
            if self.current_ee_pose is None:
                return
                
            marker_array = MarkerArray()
            
            # Determine target position based on planning mode
            if self.planning_mode == 'goal':
                target_pos = self.goal_position.copy()
            else:  # z-plane mode
                target_pos = np.array([self.current_ee_pose[0], self.current_ee_pose[1], self.target_z_plane])
            
            # Orange initial trajectory line
            line_marker = Marker()
            line_marker.header.frame_id = "iiwa_link_0"
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "initial_trajectory"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.006  # Medium thickness
            line_marker.color = ColorRGBA(1.0, 0.65, 0.0, 0.8)  # Orange with good visibility
            
            # Create joint-interpolated trajectory (like the actual initial guess)
            try:
                # Get current joint positions
                joint_positions = self.current_joint_state[:7] if self.current_joint_state is not None else np.zeros(7)
                
                # Solve IK for target position
                if hasattr(self.planner, 'kinematics'):
                    try:
                        q_target = self.planner.kinematics.solve_ik(target_pos, joint_positions)
                    except:
                        # If IK fails, fall back to straight line
                        q_target = joint_positions.copy()
                else:
                    q_target = joint_positions.copy()
                
                # Interpolate in joint space with S-curve (matching initial guess generation)
                steps = 25
                for i in range(steps + 1):
                    # S-curve interpolation (same as in initial_trajectory.py)
                    t = i / steps
                    s = 0.5 * (1 - np.cos(np.pi * t))
                    q_interp = joint_positions + s * (q_target - joint_positions)
                    
                    # Forward kinematics to get EE position
                    if hasattr(self.planner, 'kinematics'):
                        try:
                            ee_pos = self.planner.kinematics.forward_kinematics_numeric(q_interp)
                            point = Point()
                            point.x = ee_pos[0]
                            point.y = ee_pos[1]
                            point.z = ee_pos[2]
                            line_marker.points.append(point)
                        except:
                            # Fallback to linear interpolation if FK fails
                            interp_pos = (1 - t) * self.current_ee_pose + t * target_pos
                            point = Point()
                            point.x = interp_pos[0]
                            point.y = interp_pos[1]
                            point.z = interp_pos[2]
                            line_marker.points.append(point)
                    else:
                        # Fallback to linear interpolation
                        interp_pos = (1 - t) * self.current_ee_pose + t * target_pos
                        point = Point()
                        point.x = interp_pos[0]
                        point.y = interp_pos[1]
                        point.z = interp_pos[2]
                        line_marker.points.append(point)
                        
            except Exception as e:
                rospy.logwarn(f"Joint interpolation failed, using linear: {e}")
                # Fallback to straight-line interpolation
                steps = 25
                for i in range(steps + 1):
                    t = i / steps
                    interp_pos = (1 - t) * self.current_ee_pose + t * target_pos
                    
                    point = Point()
                    point.x = interp_pos[0]
                    point.y = interp_pos[1]
                    point.z = interp_pos[2]
                    line_marker.points.append(point)
            
            marker_array.markers.append(line_marker)
            
            # Add small orange spheres at start and end of initial trajectory
            # Start marker
            start_marker = Marker()
            start_marker.header.frame_id = "iiwa_link_0"
            start_marker.header.stamp = rospy.Time.now()
            start_marker.ns = "initial_trajectory"
            start_marker.id = 1
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD
            
            start_marker.pose.position.x = self.current_ee_pose[0]
            start_marker.pose.position.y = self.current_ee_pose[1]
            start_marker.pose.position.z = self.current_ee_pose[2]
            start_marker.pose.orientation.w = 1.0
            
            start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.015
            start_marker.color = ColorRGBA(1.0, 0.65, 0.0, 0.9)  # Orange
            marker_array.markers.append(start_marker)
            
            # End marker
            end_marker = Marker()
            end_marker.header.frame_id = "iiwa_link_0"
            end_marker.header.stamp = rospy.Time.now()
            end_marker.ns = "initial_trajectory"
            end_marker.id = 2
            end_marker.type = Marker.SPHERE
            end_marker.action = Marker.ADD
            
            end_marker.pose.position.x = target_pos[0]
            end_marker.pose.position.y = target_pos[1]
            end_marker.pose.position.z = target_pos[2]
            end_marker.pose.orientation.w = 1.0
            
            end_marker.scale.x = end_marker.scale.y = end_marker.scale.z = 0.015
            end_marker.color = ColorRGBA(1.0, 0.65, 0.0, 0.9)  # Orange
            marker_array.markers.append(end_marker)
            
            self.initial_traj_pub.publish(marker_array)
            rospy.logdebug(f"📊 Initial trajectory visualized: {self.planning_mode} mode, {len(line_marker.points)} points")
            
        except Exception as e:
            rospy.logwarn(f"Initial trajectory visualization failed: {e}")
            import traceback
            rospy.logwarn(f"Initial trajectory traceback: {traceback.format_exc()}")
    
    def _clear_initial_trajectory(self):
        """Clear initial trajectory markers."""
        try:
            marker_array = MarkerArray()
            
            # Delete markers by setting action to DELETE
            for i in range(3):  # Clear line + 2 spheres
                delete_marker = Marker()
                delete_marker.header.frame_id = "iiwa_link_0"
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.ns = "initial_trajectory"
                delete_marker.id = i
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
            
            self.initial_traj_pub.publish(marker_array)
            rospy.loginfo("🧹 Initial trajectory cleared for optimization")
            
        except Exception as e:
            rospy.logwarn(f"Failed to clear initial trajectory: {e}")
    
    def _visualize_goal(self):
        """Visualize goal position and target width."""
        try:
            # Goal marker
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
            
            scale = self.goal_width * self.target_scale
            goal_marker.scale.x = scale
            goal_marker.scale.y = scale
            goal_marker.scale.z = scale
            goal_marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # Red
            
            self.goal_pub.publish(goal_marker)
            
            # Target width marker
            target_marker = Marker()
            target_marker.header.frame_id = "iiwa_link_0"
            target_marker.header.stamp = rospy.Time.now()
            target_marker.ns = "target_width"
            target_marker.id = 0
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            
            target_marker.pose.position.x = self.goal_position[0]
            target_marker.pose.position.y = self.goal_position[1]
            target_marker.pose.position.z = self.goal_position[2]
            target_marker.pose.orientation.w = 1.0
            
            target_marker.scale.x = self.goal_width * 2
            target_marker.scale.y = self.goal_width * 2
            target_marker.scale.z = self.goal_width * 2
            target_marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.25)  # Transparent red
            
            self.target_pub.publish(target_marker)
            
            # Show initial trajectory when visualizing goal
            self._visualize_initial_trajectory()
            
        except Exception as e:
            rospy.logwarn(f"Goal visualization failed: {e}")
    
    def _visualize_z_plane(self):
        """Visualize target z-plane and projected target."""
        try:
            # Z-plane marker
            plane_marker = Marker()
            plane_marker.header.frame_id = "iiwa_link_0"
            plane_marker.header.stamp = rospy.Time.now()
            plane_marker.ns = "z_plane"
            plane_marker.id = 0
            plane_marker.type = Marker.CUBE
            plane_marker.action = Marker.ADD
            
            plane_marker.pose.position.x = 0.4  # Center in workspace
            plane_marker.pose.position.y = 0.0
            plane_marker.pose.position.z = self.target_z_plane
            plane_marker.pose.orientation.w = 1.0
            
            plane_marker.scale.x = self.z_plane_size
            plane_marker.scale.y = self.z_plane_size
            plane_marker.scale.z = 0.001  # Very thin plane
            plane_marker.color = ColorRGBA(0.0, 0.0, 1.0, self.z_plane_alpha)  # Blue
            
            self.z_plane_pub.publish(plane_marker)
            
            # Visualize projected target if EE pose is available
            if self.current_ee_pose is not None:
                projected_goal = self.current_ee_pose.copy()
                projected_goal[2] = self.target_z_plane
                
                # Goal marker on z-plane
                goal_marker = Marker()
                goal_marker.header.frame_id = "iiwa_link_0"
                goal_marker.header.stamp = rospy.Time.now()
                goal_marker.ns = "goal"
                goal_marker.id = 0
                goal_marker.type = Marker.SPHERE
                goal_marker.action = Marker.ADD
                
                goal_marker.pose.position.x = projected_goal[0]
                goal_marker.pose.position.y = projected_goal[1]
                goal_marker.pose.position.z = projected_goal[2]
                goal_marker.pose.orientation.w = 1.0
                
                scale = self.goal_width * self.target_scale
                goal_marker.scale.x = scale
                goal_marker.scale.y = scale
                goal_marker.scale.z = scale
                goal_marker.color = ColorRGBA(1.0, 0.5, 0.0, 0.8)  # Orange
                
                self.goal_pub.publish(goal_marker)
                
                # Target width marker on z-plane
                target_marker = Marker()
                target_marker.header.frame_id = "iiwa_link_0"
                target_marker.header.stamp = rospy.Time.now()
                target_marker.ns = "target_width"
                target_marker.id = 0
                target_marker.type = Marker.SPHERE
                target_marker.action = Marker.ADD
                
                target_marker.pose.position.x = projected_goal[0]
                target_marker.pose.position.y = projected_goal[1]
                target_marker.pose.position.z = projected_goal[2]
                target_marker.pose.orientation.w = 1.0
                
                target_marker.scale.x = self.goal_width * 2
                target_marker.scale.y = self.goal_width * 2
                target_marker.scale.z = self.goal_width * 2
                target_marker.color = ColorRGBA(1.0, 0.5, 0.0, 0.25)  # Transparent orange
                
                self.target_pub.publish(target_marker)
                
                # Show initial trajectory when visualizing z-plane
                self._visualize_initial_trajectory()
            
        except Exception as e:
            rospy.logwarn(f"Z-plane visualization failed: {e}")
    
    def _clear_all_visualizations(self):
        """Clear all visualization markers."""
        try:
            # Clear trajectory
            empty_array = MarkerArray()
            self.trajectory_pub.publish(empty_array)
            self.initial_traj_pub.publish(empty_array)
            
            # Clear other markers
            for pub in [self.goal_pub, self.z_plane_pub, self.target_pub]:
                delete_marker = Marker()
                delete_marker.header.frame_id = "iiwa_link_0"
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.action = Marker.DELETE
                pub.publish(delete_marker)
                
        except Exception as e:
            rospy.logwarn(f"Failed to clear visualizations: {e}")
    
    def _interactive_set_goal(self):
        """Set goal position interactively."""
        print(f"\n🎯 Current goal: [{self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}, {self.goal_position[2]:.3f}]")
        print("Enter new goal position:")
        
        new_goal = []
        for i, (label, current) in enumerate(zip(['X', 'Y', 'Z'], self.goal_position)):
            while True:
                try:
                    user_input = self._safe_input(f"{label} [{current:.3f}]: ").strip()
                    if user_input == "":
                        new_goal.append(current)
                        break
                    else:
                        new_goal.append(float(user_input))
                        break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        self.goal_position = np.array(new_goal)
        self._visualize_current_mode()
        self._visualize_initial_trajectory()  # Show initial trajectory immediately
        print(f"✅ Goal updated to: [{self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}, {self.goal_position[2]:.3f}]")
        rospy.loginfo("📊 Goal visualization updated in RViz")
    
    def _interactive_set_width(self):
        """Set target width interactively."""
        print(f"\n🎯 Current target width: {self.goal_width:.4f}m")
        
        while True:
            try:
                user_input = self._safe_input(f"Enter new width [{self.goal_width:.4f}]: ").strip()
                if user_input == "":
                    break
                else:
                    self.goal_width = float(user_input)
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        self._visualize_current_mode()
        self._visualize_initial_trajectory()  # Update initial trajectory with new width
        print(f"✅ Target width updated to: {self.goal_width:.4f}m")
        rospy.loginfo("📊 Target width visualization updated in RViz")
    
    def _interactive_set_z_plane(self):
        """Set z-plane height interactively."""
        print(f"\n🎯 Current z-plane height: {self.target_z_plane:.3f}m")
        
        while True:
            try:
                user_input = self._safe_input(f"Enter new z-plane height [{self.target_z_plane:.3f}]: ").strip()
                if user_input == "":
                    break
                else:
                    self.target_z_plane = float(user_input)
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        self._visualize_current_mode()
        self._visualize_initial_trajectory()  # Update initial trajectory to new z-plane
        print(f"✅ Z-plane height updated to: {self.target_z_plane:.3f}m")
        rospy.loginfo("📊 Z-plane visualization updated in RViz")
    
    def print_help(self):
        """Print help information."""
        print("\n" + "="*60)
        print("🚀 Enhanced IIWA Planning Interface - Help")
        print("="*60)
        print("Planning Modes:")
        print("  1 - Switch to goal planning mode")
        print("  2 - Switch to z-plane planning mode") 
        print(f"      Current: {self.planning_mode}")
        print("")
        
        if self.planning_mode == 'goal':
            print("Goal Planning:")
            print("  g - Set goal position")
            print("  w - Set target width")
        else:
            print("Z-Plane Planning:")
            print("  z - Set z-plane height") 
            print("  w - Set target width")
        
        print("")
        print("Planning & Execution:")
        print("  p - Plan humanlike trajectory")
        print("  c - Confirm planned trajectory")
        print("  e - Execute confirmed trajectory")
        print("")
        print("Other:")
        print("  k - Update cartesian impedance stiffness")
        print("  v - Clear visualizations")
        print("  s - Show status")
        print("  h - Show help")
        print("  q - Return to planning options / Quit")
        print("="*60)
    
    def print_status(self):
        """Print current status."""
        print("\n" + "="*50)
        print("📊 Enhanced Planning Interface Status")
        print("="*50)
        print(f"Environment: {'Simulation' if self.use_simulation else 'Real Robot'}")
        print(f"Planning mode: {self.planning_mode}")
        print(f"Controller mode: {self.controller_manager.get_current_mode()}")
        
        if self.planning_mode == 'goal':
            print(f"Goal: [{self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}, {self.goal_position[2]:.3f}]")
        else:
            print(f"Z-plane: {self.target_z_plane:.3f}m")
        
        print(f"Target width: {self.goal_width:.4f}m")
        
        if self.current_ee_pose is not None:
            print(f"Current EE: [{self.current_ee_pose[0]:.3f}, {self.current_ee_pose[1]:.3f}, {self.current_ee_pose[2]:.3f}]")
        
        traj_status = "Ready" if self.trajectory_confirmed else "Not confirmed" if self.last_planned_trajectory else "None"
        print(f"Trajectory: {traj_status}")
        print("="*50)
    
    def run_interface(self):
        """Run the main enhanced interface."""
        print("\n🚀 Enhanced IIWA Planning Interface Started!")
        print(f"Environment: {'Simulation' if self.use_simulation else 'Real Robot'}")
        
        # Show planning mode selection
        self._show_planning_options()
        
        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            new_settings = old_settings.copy()
            new_settings[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
            
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'q':
                        # Return to planning options or quit
                        choice = self._safe_input("\nReturn to planning options (r) or Quit (q)? ").lower()
                        if choice == 'r':
                            self._show_planning_options()
                            continue
                        else:
                            break
                    elif key == 'h':
                        self.print_help()
                    elif key == 's':
                        self.print_status()
                    elif key == '1':
                        self.switch_planning_mode('goal')
                    elif key == '2':
                        self.switch_planning_mode('z_plane')
                    elif key == 'g' and self.planning_mode == 'goal':
                        self._interactive_set_goal()
                    elif key == 'z' and self.planning_mode == 'z_plane':
                        self._interactive_set_z_plane()
                    elif key == 'w':
                        self._interactive_set_width()
                    elif key == 'p':
                        print("\n🎯 Planning humanlike trajectory...")
                        if self.plan_trajectory():
                            print("✅ Trajectory planned! Press 'c' to confirm, then 'e' to execute")
                            if self.last_planned_trajectory:
                                self._visualize_trajectory(self.last_planned_trajectory)
                        else:
                            print("❌ Planning failed!")
                    elif key == 'c':
                        if self.last_planned_trajectory:
                            self.trajectory_confirmed = True
                            print("✅ Trajectory confirmed! Press 'e' to execute")
                        else:
                            print("❌ No trajectory to confirm")
                    elif key == 'e':
                        if self.execute_trajectory():
                            print("✅ Execution successful!")
                        else:
                            print("❌ Execution failed!")
                    elif key == 'v':
                        self._clear_all_visualizations()
                        self._visualize_current_mode()
                        print("✅ Visualizations cleared and refreshed!")
                    elif key == 'k':
                        self.update_impedance_stiffness()
                
                # Periodic visualization updates (less frequent, no initial trajectory spam)
                if hasattr(self, '_last_viz_update'):
                    if time.time() - self._last_viz_update > 5.0:  # Every 5 seconds instead of 2
                        self._visualize_current_mode()  # Only goal/z-plane, no initial trajectory
                        self._last_viz_update = time.time()
                else:
                    self._last_viz_update = time.time()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if old_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def _show_planning_options(self):
        """Show the two planning options."""
        print("\n" + "="*60)
        print("🎯 PLANNING OPTIONS")
        print("="*60)
        print("1. 📍 GOAL PLANNING - Plan to specific 3D position")
        print("2. 🔷 Z-PLANE PLANNING - Plan to target z-height plane")
        print("")
        print("Press '1' or '2' to select planning mode")
        print("Press 'h' for help, 'q' to quit")
        print("="*60)


def main():
    """Main function."""
    try:
        interface = EnhancedIIWAPlanningInterface()
        interface.run_interface()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

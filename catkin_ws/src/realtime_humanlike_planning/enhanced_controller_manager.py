#!/usr/bin/env python3
"""
Enhanced Controller Manager for IIWA Robot
Supports both simulation and real robot with different control modes:
- Simulation: Standard impedance control for trajectory execution
- Real Robot: Native KUKA hand-guiding integration + impedance control
"""

import rospy
import time
import yaml
import os
from typing import Dict, Any, Optional
from controller_manager_msgs.srv import SwitchController, LoadController, ListControllers
from std_msgs.msg import Float64MultiArray
from dynamic_reconfigure.client import Client


class EnhancedControllerManager:
    """Enhanced controller manager with dual-mode support for sim and real robot."""
    
    def __init__(self, robot_name: str = "iiwa", use_simulation: bool = True):
        """Initialize the enhanced controller manager."""
        self.robot_name = robot_name
        self.use_simulation = use_simulation
        self.current_mode = None
        
        # Load configuration
        self._load_config()
        
        # Initialize ROS services
        self._init_services()
        
        # Define control modes based on environment
        self._define_control_modes()
        
        rospy.loginfo(f"Enhanced Controller Manager initialized for {'simulation' if use_simulation else 'real robot'}")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config', 'planning_config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            rospy.loginfo("✅ Configuration loaded from YAML file")
        except Exception as e:
            rospy.logwarn(f"Failed to load configuration: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration values."""
        return {
            'initial_cartesian_stiffness': [0.1, 0.1, 300, 300, 300, 300],
            'initial_cartesian_damping': [2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
        }
    
    def _init_services(self):
        """Initialize ROS services for controller management."""
        service_prefix = f'/{self.robot_name}/controller_manager'
        
        # Wait for controller manager services
        rospy.wait_for_service(f'{service_prefix}/switch_controller')
        rospy.wait_for_service(f'{service_prefix}/load_controller')
        rospy.wait_for_service(f'{service_prefix}/list_controllers')
        
        # Create service proxies
        self.switch_srv = rospy.ServiceProxy(f'{service_prefix}/switch_controller', SwitchController)
        self.load_srv = rospy.ServiceProxy(f'{service_prefix}/load_controller', LoadController)
        self.list_srv = rospy.ServiceProxy(f'{service_prefix}/list_controllers', ListControllers)
        
        rospy.loginfo("✅ Controller manager services connected")
    
    def _define_control_modes(self):
        """Define control modes based on simulation vs real robot."""
        stiffness = self.config.get('initial_cartesian_stiffness', [0.1, 0.1, 300, 300, 300, 300])
        damping = self.config.get('initial_cartesian_damping', [2.0, 2.0, 2.0, 2.0, 1.0, 1.0])
        
        if self.use_simulation:
            self.control_modes = {
                "impedance_control": {
                    "controller": "CartesianImpedanceController",
                    "description": "Cartesian impedance control",
                    "stiffness": stiffness,
                    "damping": damping
                }
            }
        else:
            # Real robot modes with KUKA hand-guiding integration
            self.control_modes = {
                "hand_guiding": {
                    "controller": "TorqueController",
                    "description": "Native KUKA hand-guiding with force feedback",
                    "method": "kuka_handguiding",
                    "force_threshold": [10, 10, 10, 5, 5, 5]  # N and Nm
                },
                "impedance_control": {
                    "controller": "CartesianImpedanceController",
                    "description": "Cartesian impedance control",
                    "stiffness": stiffness,
                    "damping": damping
                }
            }
        
        # Ensure joint_state_controller is always running for planning
        self._ensure_essential_controllers()
    
    def _ensure_essential_controllers(self):
        """Ensure essential controllers like joint_state_controller are always running."""
        try:
            essential_controller = f"/{self.robot_name}/joint_state_controller"
            
            # Load and start the joint state controller
            self._ensure_controller_loaded(essential_controller)
            
            # Check if it's running
            controllers_response = self.list_srv()
            joint_state_controller = None
            for c in controllers_response.controller:
                if "joint_state_controller" in c.name:
                    joint_state_controller = c
                    break
            
            if joint_state_controller and joint_state_controller.state != "running":
                rospy.loginfo("🔧 Starting joint_state_controller...")
                result = self.switch_srv(
                    start_controllers=[joint_state_controller.name],
                    stop_controllers=[],
                    strictness=1,
                    start_asap=False,
                    timeout=10.0
                )
                if result.ok:
                    rospy.loginfo("✅ joint_state_controller started successfully")
                else:
                    rospy.logwarn("❌ Failed to start joint_state_controller")
            elif joint_state_controller:
                rospy.loginfo("✅ joint_state_controller already running")
                
        except Exception as e:
            rospy.logwarn(f"Failed to ensure essential controllers: {e}")
    
    def switch_to_mode(self, mode_name: str) -> bool:
        """Switch to a specific control mode."""
        if mode_name not in self.control_modes:
            rospy.logerr(f"Unknown control mode: {mode_name}")
            return False
        
        mode_config = self.control_modes[mode_name]
        rospy.loginfo(f"🔄 Switching to {mode_name}: {mode_config['description']}")
        
        try:
            if self.use_simulation:
                return self._switch_simulation_mode(mode_name, mode_config)
            else:
                return self._switch_real_robot_mode(mode_name, mode_config)
        except Exception as e:
            rospy.logerr(f"Failed to switch to {mode_name}: {e}")
            return False
    
    def _switch_simulation_mode(self, mode_name: str, config: Dict[str, Any]) -> bool:
        """Handle simulation-specific mode switching."""
        controller_name = config["controller"]
        
        # Load required controllers
        self._ensure_controller_loaded(controller_name)
        
        # Switch to the controller
        if self._switch_controller(controller_name):
            # Configure impedance parameters if applicable
            if "stiffness" in config:
                time.sleep(1.0)  # Give controller time to initialize
                self._configure_impedance_parameters(config["stiffness"], config["damping"])
            
            self.current_mode = mode_name
            rospy.loginfo(f"✅ Successfully switched to {mode_name} (simulation)")
            return True
        
        return False
    
    def _switch_real_robot_mode(self, mode_name: str, config: Dict[str, Any]) -> bool:
        """Handle real robot-specific mode switching."""
        if mode_name == "hand_guiding":
            return self._activate_hand_guiding_mode(config)
        else:
            # Standard controller switching for impedance control
            controller_name = config["controller"]
            self._ensure_controller_loaded(controller_name)
            
            if self._switch_controller(controller_name):
                if "stiffness" in config:
                    time.sleep(1.0)
                    self._configure_impedance_parameters(config["stiffness"], config["damping"])
                
                self.current_mode = mode_name
                rospy.loginfo(f"✅ Successfully switched to {mode_name} (real robot)")
                return True
        
        return False
    
    def _activate_hand_guiding_mode(self, config: Dict[str, Any]) -> bool:
        """Activate KUKA hand-guiding mode for real robot."""
        rospy.loginfo("🤖 Activating KUKA hand-guiding mode...")
        
        if self._switch_controller("TorqueController"):
            # Publish zero torques to enable free movement
            self._publish_zero_torques()
            
            self.current_mode = "hand_guiding"
            rospy.loginfo("✅ Hand-guiding mode activated")
            return True
        
        return False
    
    def _switch_controller(self, target_controller: str) -> bool:
        """Switch to a specific controller."""
        try:
            # Get current active controllers
            controllers_response = self.list_srv()
            active_controllers = [c.name for c in controllers_response.controller if c.state == "running"]
            
            # Don't switch if already active
            if target_controller in active_controllers:
                rospy.loginfo(f"Controller {target_controller} already active")
                return True
            
            # IMPORTANT: Don't stop joint_state_controller - it's needed for planning!
            controllers_to_stop = [c for c in active_controllers 
                                 if "joint_state_controller" not in c and c != target_controller]
            
            rospy.loginfo(f"Starting: [{target_controller}], Stopping: {controllers_to_stop}")
            
            # Switch controllers
            result = self.switch_srv(
                start_controllers=[target_controller],
                stop_controllers=controllers_to_stop,
                strictness=1,  # BEST_EFFORT
                start_asap=False,
                timeout=10.0
            )
            
            return result.ok
            
        except Exception as e:
            rospy.logerr(f"Failed to switch to {target_controller}: {e}")
            return False
    
    def _ensure_controller_loaded(self, controller_name: str):
        """Ensure a controller is loaded."""
        try:
            controllers_response = self.list_srv()
            existing_controllers = [c.name for c in controllers_response.controller]
            
            if controller_name in existing_controllers:
                rospy.loginfo(f"Controller {controller_name} already loaded")
                return
            
            rospy.loginfo(f"Loading controller: {controller_name}")
            self.load_srv(controller_name)
            time.sleep(0.5)
            rospy.loginfo(f"✅ Controller {controller_name} loaded successfully")
        except Exception as e:
            rospy.logwarn(f"Failed to load {controller_name}: {e}")
    
    def _configure_impedance_parameters(self, stiffness: list, damping: list):
        """Configure impedance parameters using dynamic reconfigure."""
        try:
            rospy.loginfo(f"📊 Configuring impedance: stiffness={stiffness}")
            
            if len(stiffness) == 6:
                try:
                    client = Client("/iiwa/CartesianImpedanceController/dynamic_reconfigure_server_node", timeout=15)
                    time.sleep(2.0)
                    
                    config = {
                        "seperate_axis": True,
                        "translational_stiffness_x": stiffness[0],
                        "translational_stiffness_y": stiffness[1], 
                        "translational_stiffness_z": stiffness[2],
                        "rotational_stiffness_alpha": stiffness[3],
                        "rotational_stiffness_theta": stiffness[4],
                        "rotational_stiffness_phi": stiffness[5],
                        "translational_damping_ratio": 0.7,
                        "rotational_damping_ratio": 0.7
                    }
                    
                    client.update_configuration(config)
                    rospy.loginfo("✅ CartesianImpedanceController configured successfully!")
                    
                except Exception as e:
                    rospy.logwarn(f"Failed to configure CartesianImpedanceController: {e}")
            
        except Exception as e:
            rospy.logwarn(f"Failed to configure impedance parameters: {e}")
    
    def _publish_zero_torques(self):
        """Publish zero torques for hand-guiding mode."""
        try:
            torque_pub = rospy.Publisher(f'/{self.robot_name}/TorqueController/command', 
                                       Float64MultiArray, queue_size=1)
            time.sleep(0.5)
            
            zero_torques = Float64MultiArray()
            zero_torques.data = [0.0] * 7
            
            for _ in range(10):
                torque_pub.publish(zero_torques)
                time.sleep(0.1)
                
            rospy.loginfo("🔄 Published zero torques for free movement")
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish zero torques: {e}")
    
    def update_stiffness_parameters(self, stiffness: list, damping: Optional[list] = None) -> bool:
        """Update stiffness parameters directly."""
        try:
            if damping is None:
                damping = self.config.get('initial_cartesian_damping', [2.0, 2.0, 2.0, 2.0, 1.0, 1.0])
            
            rospy.loginfo(f"🔧 Updating stiffness parameters: {stiffness}")
            self._configure_impedance_parameters(stiffness, damping)
            return True
        except Exception as e:
            rospy.logerr(f"Failed to update stiffness parameters: {e}")
            return False
    
    def get_current_mode(self) -> Optional[str]:
        """Get the current control mode."""
        return self.current_mode
    
    def get_available_modes(self) -> Dict[str, str]:
        """Get available control modes with descriptions."""
        return {mode: config["description"] for mode, config in self.control_modes.items()}


def main():
    """Main function for testing the enhanced controller manager."""
    import sys
    
    rospy.init_node('enhanced_controller_manager', anonymous=True)
    
    robot_name = str(rospy.get_param('~robot_name', 'iiwa'))
    use_simulation = bool(rospy.get_param('~use_simulation', True))
    
    manager = EnhancedControllerManager(robot_name, use_simulation)
    
    # Start in appropriate initial mode
    initial_mode = "impedance_control" if use_simulation else "hand_guiding"
    manager.switch_to_mode(initial_mode)
    
    rospy.loginfo("Enhanced Controller Manager ready!")
    rospy.spin()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script to set the robot to a good initial configuration.
"""

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import SwitchController
import time

class InitialConfigSetter:
    def __init__(self):
        rospy.init_node('set_initial_config', anonymous=True)
        
        # Target configuration (positions only - first 7 elements)
        self.target_config = np.array([
            1.72881887e-06,  2.80260657e-01,  1.59761569e-06, -1.53136477e+00,
            -7.90098921e-06,  1.31590271e+00,  0.00000000e+00
        ])
        
        # Current joint positions
        self.current_positions = None
        
        # Publishers and subscribers
        self.joint_state_sub = rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_state_callback)
        
        # Service client for controller switching
        try:
            rospy.wait_for_service('/iiwa/controller_manager/switch_controller', timeout=5.0)
            self.switch_controller = rospy.ServiceProxy('/iiwa/controller_manager/switch_controller', SwitchController)
        except:
            rospy.logwarn("Controller manager service not available")
            self.switch_controller = None
    
    def joint_state_callback(self, msg):
        """Store current joint positions."""
        expected_joints = [
            'iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4',
            'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7'
        ]
        
        positions = np.zeros(7)
        for i, joint_name in enumerate(expected_joints):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                positions[i] = msg.position[idx]
        
        self.current_positions = positions
    
    def switch_to_position_controller(self):
        """Switch to position controller."""
        if self.switch_controller:
            try:
                # First stop the CartesianImpedanceController
                response = self.switch_controller(
                    start_controllers=['/iiwa/PositionController'],
                    stop_controllers=['CartesianImpedanceController'],
                    strictness=1,
                    start_asap=False,
                    timeout=rospy.Duration(5.0)
                )
                if response.ok:
                    rospy.loginfo("✅ Switched to position controller")
                    return True
                else:
                    rospy.logerr("❌ Failed to switch to position controller")
                    return False
            except Exception as e:
                rospy.logerr(f"Failed to call controller switch service: {e}")
                return False
        return False
    
    def switch_back_to_impedance_controller(self):
        """Switch back to impedance controller."""
        if self.switch_controller:
            try:
                response = self.switch_controller(
                    start_controllers=['CartesianImpedanceController'],
                    stop_controllers=['/iiwa/PositionController'],
                    strictness=1,
                    start_asap=False,
                    timeout=rospy.Duration(5.0)
                )
                if response.ok:
                    rospy.loginfo("✅ Switched back to impedance controller")
                    return True
                else:
                    rospy.logerr("❌ Failed to switch back to impedance controller")
                    return False
            except Exception as e:
                rospy.logerr(f"Failed to switch back to impedance controller: {e}")
                return False
        return False
    
    def set_initial_configuration(self):
        """Set robot to the target initial configuration."""
        rospy.loginfo("Setting robot to initial configuration...")
        
        # Wait for joint states
        while self.current_positions is None and not rospy.is_shutdown():
            rospy.loginfo("Waiting for joint states...")
            rospy.sleep(0.5)
        
        if self.current_positions is None:
            rospy.logerr("No joint states received")
            return False
        
        rospy.loginfo(f"Current position: {self.current_positions}")
        rospy.loginfo(f"Target position: {self.target_config}")
        
        # Check if already in target position
        if np.allclose(self.current_positions, self.target_config, atol=0.05):
            rospy.loginfo("✅ Already in target configuration")
            return True
        
        # Switch to position controller
        if not self.switch_to_position_controller():
            rospy.logerr("Failed to switch to position controller")
            return False
        
        # Wait for controller to start
        rospy.sleep(2.0)
        
        # Create position publisher
        pos_pub = rospy.Publisher('/iiwa/PositionController/command', Float64MultiArray, queue_size=1)
        rospy.sleep(1.0)  # Wait for publisher to connect
        
        # Send target position command
        cmd = Float64MultiArray()
        cmd.data = self.target_config.tolist()
        
        rospy.loginfo("Moving to target configuration...")
        
        # Send command repeatedly until reached
        rate = rospy.Rate(10)  # 10 Hz
        start_time = time.time()
        timeout = 15.0  # 15 seconds timeout
        
        while not rospy.is_shutdown() and (time.time() - start_time) < timeout:
            pos_pub.publish(cmd)
            
            if self.current_positions is not None:
                error = np.linalg.norm(self.current_positions - self.target_config)
                if error < 0.05:  # 0.05 rad tolerance
                    rospy.loginfo("✅ Reached target configuration!")
                    
                    # Switch back to impedance controller
                    rospy.sleep(1.0)
                    success = self.switch_back_to_impedance_controller()
                    
                    if success:
                        rospy.loginfo("🎉 Robot is now in good initial configuration with impedance controller active!")
                    else:
                        rospy.logwarn("Robot reached target but failed to switch back to impedance controller")
                    
                    return success
                
                rospy.loginfo(f"Moving... error: {error:.3f} rad")
            
            rate.sleep()
        
        rospy.logwarn("Timeout - failed to reach target configuration")
        # Try to switch back anyway
        self.switch_back_to_impedance_controller()
        return False

def main():
    try:
        setter = InitialConfigSetter()
        success = setter.set_initial_configuration()
        
        if success:
            rospy.loginfo("✅ Initial configuration set successfully!")
            rospy.loginfo("🚀 You can now use the planning interface without KDL errors!")
        else:
            rospy.logerr("❌ Failed to set initial configuration")
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    main()

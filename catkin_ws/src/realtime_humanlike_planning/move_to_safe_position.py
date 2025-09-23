#!/usr/bin/env python3
"""
Simple script to move IIWA robot to a safe configuration using position control.
This avoids the KDL errors that occur with impedance control at zero configuration.
"""

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import SwitchController
import time

class SafePositionMover:
    def __init__(self):
        rospy.init_node('safe_position_mover', anonymous=True)
        
        # Safe configuration (not all zeros)
        self.safe_config = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
        
        # Publishers for position control
        self.pos_pub = rospy.Publisher('/iiwa/PositionController/command', Float64MultiArray, queue_size=1)
        
        # Subscriber for joint states
        self.joint_state_sub = rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_state_callback)
        self.current_positions = None
        
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
                response = self.switch_controller(
                    start_controllers=['PositionController'],
                    stop_controllers=['TorqueController'],
                    strictness=2
                )
                if response.ok:
                    rospy.loginfo("Switched to position controller")
                    return True
                else:
                    rospy.logerr("Failed to switch to position controller")
                    return False
            except Exception as e:
                rospy.logerr(f"Failed to call controller switch service: {e}")
                return False
        return False
    
    def move_to_safe_position(self):
        """Move robot to safe configuration."""
        rospy.loginfo("Moving robot to safe position...")
        
        # Wait for joint states
        while self.current_positions is None and not rospy.is_shutdown():
            rospy.loginfo("Waiting for joint states...")
            rospy.sleep(0.5)
        
        if self.current_positions is None:
            rospy.logerr("No joint states received")
            return False
        
        rospy.loginfo(f"Current position: {self.current_positions}")
        rospy.loginfo(f"Target position: {self.safe_config}")
        
        # Check if already in safe position
        if np.allclose(self.current_positions, self.safe_config, atol=0.1):
            rospy.loginfo("Already in safe position")
            return True
        
        # Switch to position controller
        if not self.switch_to_position_controller():
            rospy.logerr("Failed to switch to position controller")
            return False
        
        # Wait a bit for controller to start
        rospy.sleep(1.0)
        
        # Send safe position command
        cmd = Float64MultiArray()
        cmd.data = self.safe_config.tolist()
        
        rospy.loginfo("Sending safe position command...")
        
        # Send command repeatedly until reached
        rate = rospy.Rate(10)  # 10 Hz
        start_time = time.time()
        timeout = 10.0  # 10 seconds timeout
        
        while not rospy.is_shutdown() and (time.time() - start_time) < timeout:
            self.pos_pub.publish(cmd)
            
            if self.current_positions is not None:
                error = np.linalg.norm(self.current_positions - self.safe_config)
                if error < 0.05:  # 0.05 rad tolerance
                    rospy.loginfo("✅ Reached safe position!")
                    return True
                
                rospy.loginfo(f"Moving... error: {error:.3f}")
            
            rate.sleep()
        
        rospy.logwarn("Timeout or failed to reach safe position")
        return False

def main():
    try:
        mover = SafePositionMover()
        success = mover.move_to_safe_position()
        
        if success:
            rospy.loginfo("✅ Robot is now in safe configuration!")
            rospy.loginfo("You can now load the impedance controller and use the planning interface.")
        else:
            rospy.logerr("❌ Failed to move to safe configuration")
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    main()

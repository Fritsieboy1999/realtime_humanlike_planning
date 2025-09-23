#!/usr/bin/env python3
"""
Script to properly set up the impedance controller.
This handles the controller loading and switching sequence.
"""

import rospy
import time
from controller_manager_msgs.srv import LoadController, SwitchController
from std_srvs.srv import Empty

def setup_impedance_controller():
    """Set up the impedance controller properly."""
    rospy.init_node('setup_impedance_controller', anonymous=True)
    
    # Wait for controller manager services
    rospy.loginfo("Waiting for controller manager services...")
    rospy.wait_for_service('/iiwa/controller_manager/load_controller')
    rospy.wait_for_service('/iiwa/controller_manager/switch_controller')
    
    # Create service proxies
    load_controller = rospy.ServiceProxy('/iiwa/controller_manager/load_controller', LoadController)
    switch_controller = rospy.ServiceProxy('/iiwa/controller_manager/switch_controller', SwitchController)
    
    try:
        # Step 1: Load the impedance controller
        rospy.loginfo("Loading CartesianImpedanceController...")
        response = load_controller('CartesianImpedanceController')
        if response.ok:
            rospy.loginfo("✅ CartesianImpedanceController loaded successfully")
        else:
            rospy.logerr("❌ Failed to load CartesianImpedanceController")
            return False
        
        # Step 2: Wait a bit for the controller to initialize
        rospy.sleep(2.0)
        
        # Step 3: Switch from TorqueController to CartesianImpedanceController
        rospy.loginfo("Switching to CartesianImpedanceController...")
        response = switch_controller(
            start_controllers=['CartesianImpedanceController'],
            stop_controllers=['/iiwa/TorqueController'],
            strictness=2,
            start_asap=False,
            timeout=rospy.Duration(5.0)
        )
        
        if response.ok:
            rospy.loginfo("✅ Successfully switched to CartesianImpedanceController")
            rospy.loginfo("🎉 Impedance controller setup complete!")
            return True
        else:
            rospy.logerr("❌ Failed to switch to CartesianImpedanceController")
            return False
            
    except Exception as e:
        rospy.logerr(f"Error setting up impedance controller: {e}")
        return False

if __name__ == '__main__':
    try:
        success = setup_impedance_controller()
        if success:
            rospy.loginfo("Impedance controller is ready for use!")
        else:
            rospy.logerr("Failed to set up impedance controller")
    except rospy.ROSInterruptException:
        pass

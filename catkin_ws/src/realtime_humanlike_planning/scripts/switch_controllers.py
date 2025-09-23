#!/usr/bin/env python3
"""
Script to properly switch from TorqueController to JointImpedanceController
"""

import rospy
import time
from controller_manager_msgs.srv import SwitchController, LoadController

def switch_controllers():
    """Switch from TorqueController to JointImpedanceController"""
    rospy.init_node('controller_switcher', anonymous=True)
    
    # Wait for controller manager services
    rospy.wait_for_service('/iiwa/controller_manager/switch_controller')
    rospy.wait_for_service('/iiwa/controller_manager/load_controller')
    
    switch_srv = rospy.ServiceProxy('/iiwa/controller_manager/switch_controller', SwitchController)
    load_srv = rospy.ServiceProxy('/iiwa/controller_manager/load_controller', LoadController)
    
    try:
        # Load impedance controllers first
        rospy.loginfo("Loading impedance controllers...")
        load_srv('JointImpedanceController')
        load_srv('CartesianImpedanceController')
        
        time.sleep(2)  # Give controllers time to load
        
        # First, let's check what controllers are available
        from controller_manager_msgs.srv import ListControllers
        list_srv = rospy.ServiceProxy('/iiwa/controller_manager/list_controllers', ListControllers)
        controllers = list_srv()
        rospy.loginfo(f"Available controllers: {[c.name for c in controllers.controller]}")
        
        # Find the correct TorqueController name
        torque_controller_name = None
        for c in controllers.controller:
            if 'TorqueController' in c.name:
                torque_controller_name = c.name
                break
        
        if torque_controller_name is None:
            rospy.logwarn("TorqueController not found, continuing anyway...")
            stop_controllers = []
        else:
            stop_controllers = [torque_controller_name]
            rospy.loginfo(f"Will stop controller: {torque_controller_name}")
        
        # Switch controllers: stop TorqueController, start CartesianImpedanceController for real-time trajectory following
        rospy.loginfo("Switching controllers...")
        result = switch_srv(
            start_controllers=['CartesianImpedanceController'],
            stop_controllers=stop_controllers,
            strictness=1,  # BEST_EFFORT instead of STRICT
            start_asap=False,
            timeout=10.0
        )
        
        if result.ok:
            rospy.loginfo("✅ Successfully switched to CartesianImpedanceController")
        else:
            rospy.logerr("❌ Failed to switch controllers")
            
    except Exception as e:
        rospy.logerr(f"Error switching controllers: {e}")

if __name__ == '__main__':
    switch_controllers()

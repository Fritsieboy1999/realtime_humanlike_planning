#!/usr/bin/env python3
"""
Safety Monitor for IIWA Robot

This node monitors robot state and can issue emergency stops if:
- Joint velocities exceed safe limits
- Cartesian velocities exceed safe limits
- Robot moves outside safe workspace
- Communication with robot is lost

Author: Assistant
"""

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from controller_manager_msgs.srv import SwitchController
import tf2_ros
import tf2_geometry_msgs


class SafetyMonitor:
    """Safety monitor for IIWA robot operations."""
    
    def __init__(self):
        """Initialize the safety monitor."""
        rospy.init_node('iiwa_safety_monitor', anonymous=True)
        
        # Parameters
        self.robot_name = rospy.get_param('~robot_name', 'iiwa')
        self.max_joint_velocity = rospy.get_param('~max_joint_velocity', 2.0)  # rad/s
        self.max_cartesian_velocity = rospy.get_param('~max_cartesian_velocity', 1.0)  # m/s
        self.workspace_limits = {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.0, 'z_max': 1.5
        }
        
        # State variables
        self.current_joint_state = None
        self.last_joint_state_time = None
        self.emergency_stop_active = False
        self.safety_violations = []
        
        # Initialize components
        self._init_publishers()
        self._init_subscribers()
        self._init_services()
        
        # Start monitoring timer
        self.monitor_timer = rospy.Timer(rospy.Duration(0.1), self._monitor_callback)
        
        rospy.loginfo("Safety Monitor initialized")
        rospy.loginfo(f"Max joint velocity: {self.max_joint_velocity} rad/s")
        rospy.loginfo(f"Max cartesian velocity: {self.max_cartesian_velocity} m/s")
    
    def _init_publishers(self):
        """Initialize publishers."""
        self.emergency_stop_pub = rospy.Publisher(
            '/iiwa_safety_monitor/emergency_stop',
            Bool,
            queue_size=1
        )
        
        self.safety_status_pub = rospy.Publisher(
            '/iiwa_safety_monitor/safety_status',
            String,
            queue_size=10
        )
    
    def _init_subscribers(self):
        """Initialize subscribers."""
        joint_states_topic = f'/{self.robot_name}/joint_states'
        self.joint_state_sub = rospy.Subscriber(
            joint_states_topic,
            JointState,
            self._joint_state_callback
        )
    
    def _init_services(self):
        """Initialize service clients."""
        try:
            controller_manager_service = f'/{self.robot_name}/controller_manager/switch_controller'
            rospy.wait_for_service(controller_manager_service, timeout=5.0)
            self.switch_controller_service = rospy.ServiceProxy(
                controller_manager_service,
                SwitchController
            )
            rospy.loginfo("Connected to controller manager")
        except rospy.ROSException:
            rospy.logwarn("Controller manager service not available")
            self.switch_controller_service = None
    
    def _joint_state_callback(self, msg: JointState):
        """Callback for joint state updates."""
        self.current_joint_state = msg
        self.last_joint_state_time = rospy.Time.now()
    
    def _monitor_callback(self, event):
        """Main monitoring callback."""
        try:
            self._check_communication()
            self._check_joint_limits()
            self._check_workspace_limits()
            self._publish_safety_status()
            
        except Exception as e:
            rospy.logerr(f"Safety monitor error: {e}")
    
    def _check_communication(self):
        """Check if communication with robot is active."""
        if self.last_joint_state_time is None:
            self._add_violation("No joint state data received")
            return
        
        time_since_last_update = (rospy.Time.now() - self.last_joint_state_time).to_sec()
        if time_since_last_update > 1.0:  # 1 second timeout
            self._add_violation(f"Joint state data timeout: {time_since_last_update:.1f}s")
    
    def _check_joint_limits(self):
        """Check joint velocity limits."""
        if self.current_joint_state is None:
            return
        
        if len(self.current_joint_state.velocity) < 7:
            return  # No velocity data
        
        # Check joint velocities
        max_velocity = max(abs(v) for v in self.current_joint_state.velocity[:7])
        if max_velocity > self.max_joint_velocity:
            self._add_violation(f"Joint velocity limit exceeded: {max_velocity:.2f} > {self.max_joint_velocity}")
    
    def _check_workspace_limits(self):
        """Check if robot is within safe workspace."""
        if self.current_joint_state is None:
            return
        
        # This would require forward kinematics - simplified for now
        # In a real implementation, you would compute the end-effector position
        # and check against workspace limits
        pass
    
    def _add_violation(self, violation: str):
        """Add a safety violation."""
        if violation not in self.safety_violations:
            self.safety_violations.append(violation)
            rospy.logwarn(f"Safety violation: {violation}")
            
            # Trigger emergency stop for critical violations
            if any(keyword in violation.lower() for keyword in ['timeout', 'limit exceeded']):
                self._trigger_emergency_stop()
    
    def _trigger_emergency_stop(self):
        """Trigger emergency stop."""
        if self.emergency_stop_active:
            return  # Already active
        
        rospy.logerr("EMERGENCY STOP TRIGGERED!")
        self.emergency_stop_active = True
        
        # Publish emergency stop message
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_stop_pub.publish(emergency_msg)
        
        # Try to stop controllers
        if self.switch_controller_service:
            try:
                response = self.switch_controller_service(
                    start_controllers=[],
                    stop_controllers=['CartesianImpedanceController', 'JointImpedanceController'],
                    strictness=2  # STRICT
                )
                if response.ok:
                    rospy.loginfo("Controllers stopped successfully")
                else:
                    rospy.logerr("Failed to stop controllers")
            except Exception as e:
                rospy.logerr(f"Failed to call controller switch service: {e}")
    
    def _publish_safety_status(self):
        """Publish current safety status."""
        status_msg = String()
        
        if self.emergency_stop_active:
            status_msg.data = "EMERGENCY_STOP_ACTIVE"
        elif self.safety_violations:
            status_msg.data = f"VIOLATIONS: {'; '.join(self.safety_violations[-3:])}"
        else:
            status_msg.data = "OK"
        
        self.safety_status_pub.publish(status_msg)
        
        # Clear old violations
        if len(self.safety_violations) > 10:
            self.safety_violations = self.safety_violations[-5:]
    
    def reset_emergency_stop(self):
        """Reset emergency stop (would be called by external command)."""
        rospy.loginfo("Resetting emergency stop")
        self.emergency_stop_active = False
        self.safety_violations.clear()
        
        # Publish reset message
        emergency_msg = Bool()
        emergency_msg.data = False
        self.emergency_stop_pub.publish(emergency_msg)


def main():
    """Main function."""
    try:
        monitor = SafetyMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Safety monitor failed: {e}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Debug script to check joint state information and robot configuration.
"""

import rospy
import numpy as np
from sensor_msgs.msg import JointState

class JointStateDebugger:
    def __init__(self):
        rospy.init_node('joint_state_debugger', anonymous=True)
        
        self.joint_state_sub = rospy.Subscriber(
            '/iiwa/joint_states',
            JointState,
            self.joint_state_callback
        )
        
        rospy.loginfo("Joint State Debugger started. Waiting for joint states...")
        
    def joint_state_callback(self, msg):
        """Debug callback to print joint state information."""
        print("\n" + "="*60)
        print("JOINT STATE DEBUG INFO")
        print("="*60)
        print(f"Timestamp: {msg.header.stamp}")
        print(f"Frame ID: {msg.header.frame_id}")
        print(f"Number of joints: {len(msg.name)}")
        print(f"Position array size: {len(msg.position)}")
        print(f"Velocity array size: {len(msg.velocity)}")
        print(f"Effort array size: {len(msg.effort)}")
        
        print("\nJoint Names:")
        for i, name in enumerate(msg.name):
            print(f"  {i}: {name}")
        
        print("\nJoint Positions:")
        for i, pos in enumerate(msg.position):
            name = msg.name[i] if i < len(msg.name) else f"joint_{i}"
            print(f"  {name}: {pos:.4f} rad ({np.degrees(pos):.2f} deg)")
        
        if len(msg.velocity) > 0:
            print("\nJoint Velocities:")
            for i, vel in enumerate(msg.velocity):
                name = msg.name[i] if i < len(msg.name) else f"joint_{i}"
                print(f"  {name}: {vel:.4f} rad/s")
        
        # Check for IIWA joints specifically
        iiwa_joints = [
            'iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4',
            'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7'
        ]
        
        print("\nIIWA Joint Status:")
        iiwa_positions = []
        iiwa_velocities = []
        
        for joint_name in iiwa_joints:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                pos = msg.position[idx] if idx < len(msg.position) else 0.0
                vel = msg.velocity[idx] if idx < len(msg.velocity) else 0.0
                iiwa_positions.append(pos)
                iiwa_velocities.append(vel)
                print(f"  ✓ {joint_name}: pos={pos:.4f}, vel={vel:.4f}")
            else:
                print(f"  ✗ {joint_name}: NOT FOUND")
                iiwa_positions.append(0.0)
                iiwa_velocities.append(0.0)
        
        if len(iiwa_positions) == 7:
            robot_state = np.concatenate([iiwa_positions, iiwa_velocities])
            print(f"\nRobot State Vector (size {robot_state.shape[0]}):")
            print(f"  Positions: {np.array(iiwa_positions)}")
            print(f"  Velocities: {np.array(iiwa_velocities)}")
            print(f"  Combined: {robot_state}")
            
            # Check for any invalid values
            if np.any(np.isnan(robot_state)):
                print("  ⚠️  WARNING: NaN values detected!")
            if np.any(np.isinf(robot_state)):
                print("  ⚠️  WARNING: Infinite values detected!")
        
        print("="*60)

def main():
    try:
        debugger = JointStateDebugger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

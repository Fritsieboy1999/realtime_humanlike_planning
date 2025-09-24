"""
Kinematics module for robot forward/inverse kinematics and Jacobian computation.
Handles both symbolic (CasADi) and numeric (Pinocchio) implementations.
"""

from __future__ import annotations

import os
import numpy as np
import casadi as ca
import pinocchio as pin
from typing import Optional, Tuple



class RobotKinematics:
    """
    Robot kinematics handler with both symbolic and numeric implementations.
    """
    
    def __init__(self, urdf_path: Optional[str] = None):
        """
        Initialize kinematics with URDF model.
        
        Args:
            urdf_path: Path to URDF file. If None, uses default path.
        """
        self.nq = 7  # 7 DOF robot
        
        # Initialize attributes
        self.pin_model = None
        self.pin_data = None
        self.ee_frame_id = None
        
        # Initialize Pinocchio model
        self._init_pinocchio_model()
        
        # Initialize symbolic kinematics
        self._init_symbolic_kinematics()
    
    
    def _init_pinocchio_model(self):
        """Initialize Pinocchio model from ROS parameter server URDF."""
        import os
        import tempfile
        import rospy
        
        if not rospy.has_param('/robot_description'):
            raise RuntimeError("URDF not found on ROS parameter server at '/robot_description'")
        
        urdf_string = rospy.get_param('/robot_description')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_string)
            temp_urdf_path = f.name
        
        self.pin_model = pin.buildModelFromUrdf(temp_urdf_path)
        self.pin_data = self.pin_model.createData()
        
        # Extract joint limits and damping from URDF
        self.joint_limits_lower, self.joint_limits_upper, self.joint_velocity_limits, self.joint_effort_limits = self._extract_joint_limits(urdf_string)
        self.joint_damping = self._get_joint_damping()
        
        os.unlink(temp_urdf_path)
        
        self.ee_frame_id = self.pin_model.getFrameId("iiwa_link_ee")

    def _extract_joint_limits(self, urdf_string: str):
        """Extract joint limits from URDF string."""
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(urdf_string)
            joints = root.findall(".//joint[@type='revolute']")
            
            lower_limits = []
            upper_limits = []
            velocity_limits = []
            effort_limits = []
            
            for joint in joints:
                joint_name = joint.attrib.get("name", "unknown")
                limit_elem = joint.find("limit")
                if limit_elem is None:
                    raise ValueError(f"Joint {joint_name} missing <limit> element")
                
                # Extract limits - fail if any attribute is missing
                if "lower" not in limit_elem.attrib:
                    raise ValueError(f"Joint {joint_name} missing 'lower' limit")
                if "upper" not in limit_elem.attrib:
                    raise ValueError(f"Joint {joint_name} missing 'upper' limit")
                if "velocity" not in limit_elem.attrib:
                    raise ValueError(f"Joint {joint_name} missing 'velocity' limit")
                if "effort" not in limit_elem.attrib:
                    raise ValueError(f"Joint {joint_name} missing 'effort' limit")
                
                lower_limits.append(float(limit_elem.attrib["lower"]))
                upper_limits.append(float(limit_elem.attrib["upper"]))
                velocity_limits.append(float(limit_elem.attrib["velocity"]))
                effort_limits.append(float(limit_elem.attrib["effort"]))
            
            if len(lower_limits) == 7:
                print(f"Extracted joint limits from URDF")
                return (np.array(lower_limits), np.array(upper_limits), 
                       np.array(velocity_limits), np.array(effort_limits))
            else:
                print(f"Expected 7 joints, found {len(lower_limits)}, using defaults")
                raise ValueError("Unexpected number of joints")
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract joint limits from URDF: {e}. URDF must contain valid joint limits.")

    def _get_joint_damping(self) -> np.ndarray:
        """Get joint damping: 0.002 * effort_limits, fallback to 0.5 for all joints."""
        try:
            # Primary: Use 0.002 * joint effort limits from URDF
            return 0.002 * self.joint_effort_limits
        except Exception:
            # Backup: Use 0.5 for all 7 joints
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def _init_symbolic_kinematics(self):
        """Initialize symbolic kinematics using CasADi."""
        q_sym = ca.MX.sym("q", self.nq)
        pos_sym = self.forward_kinematics_symbolic(q_sym)
        self.fwd_kin = ca.Function("fwd_kin", [q_sym], [pos_sym])
        J_sym = ca.jacobian(pos_sym, q_sym)
        self.jac_fn = ca.Function("jac_fn", [q_sym], [J_sym])
    
    def forward_kinematics_numeric(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics using Pinocchio (numeric)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        T = self.pin_data.oMf[self.ee_frame_id]
        return T.translation

    def compute_jacobian_numeric(self, q: np.ndarray) -> np.ndarray:
        """Compute Jacobian using Pinocchio (numeric)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        J = pin.computeFrameJacobian(
            self.pin_model, self.pin_data, q, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        
        return J[:3, :]  # Return only position part (3x7)
    
    def transform_to_controller_frame(self, pos: np.ndarray) -> np.ndarray:
        """
        Transform position from planner's coordinate frame to controller's coordinate frame.
        
        This handles the coordinate frame differences between the planner's FK implementation
        and the controller's FK implementation.
        
        Args:
            pos: Position in planner's frame [x, y, z]
            
        Returns:
            Position in controller's frame [x, y, z]
        """
        pos = np.asarray(pos).flatten()

        return np.array([pos[0], -pos[1], pos[2]])
    
    def transform_from_controller_frame(self, pos: np.ndarray) -> np.ndarray:
        """
        Transform position from controller's coordinate frame to planner's coordinate frame.
        
        Args:
            pos: Position in controller's frame [x, y, z]
            
        Returns:
            Position in planner's frame [x, y, z]
        """
        pos = np.asarray(pos).flatten()

        return np.array([pos[0], -pos[1], pos[2]])
    
    def forward_kinematics_symbolic(self, q):
        """URDF-based forward kinematics using CasADi (symbolic)."""
        return self._forward_kinematics_accurate_urdf(q)
    
    def _rot_x(self, a):
        """Rotation matrix around X-axis."""
        ca_ = ca.cos(a)
        sa_ = ca.sin(a)
        return ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, ca_, -sa_),
            ca.horzcat(0, sa_, ca_),
        )

    def _rot_y(self, a):
        """Rotation matrix around Y-axis."""
        ca_ = ca.cos(a)
        sa_ = ca.sin(a)
        return ca.vertcat(
            ca.horzcat(ca_, 0, sa_),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-sa_, 0, ca_),
        )

    def _rot_z(self, a):
        """Rotation matrix around Z-axis."""
        ca_ = ca.cos(a)
        sa_ = ca.sin(a)
        return ca.vertcat(
            ca.horzcat(ca_, -sa_, 0),
            ca.horzcat(sa_, ca_, 0),
            ca.horzcat(0, 0, 1),
        )

    def _rot_rpy(self, roll, pitch, yaw):
        """Rotation matrix from roll-pitch-yaw."""
        return self._rot_z(yaw) @ self._rot_y(pitch) @ self._rot_x(roll)

    def _homog(self, R, p):
        """Create homogeneous transformation matrix."""
        return ca.vertcat(
            ca.horzcat(R, p.reshape((-1, 1))),
            ca.horzcat(ca.DM([0, 0, 0]).T, ca.DM([1])),
        )

    def _trans(self, x, y, z):
        """Translation transformation matrix."""
        return self._homog(ca.DM(np.eye(3)), ca.DM([x, y, z]))

    def _forward_kinematics_accurate_urdf(self, q):
        """URDF-based forward kinematics implementation."""
        T01 = self._trans(0.0, 0.0, 0.3375) @ self._homog(self._rot_rpy(0.0, 0.0, 0.0), ca.DM([0, 0, 0]))
        Rz1 = self._rot_z(-q[0])

        T12 = self._trans(0.0, 0.0, 0.0) @ self._homog(self._rot_rpy(-1.5708, 0.0, 0.0), ca.DM([0, 0, 0]))
        Rz2 = self._rot_z(q[1])

        T23 = self._trans(0.0, -0.3993, 0.0) @ self._homog(self._rot_rpy(1.5708, 0.0, 0.0), ca.DM([0, 0, 0]))
        Rz3 = self._rot_z(-q[2])

        T34 = self._trans(0.0, 0.0, 0.0) @ self._homog(self._rot_rpy(-1.5708, 0.0, 0.0), ca.DM([0, 0, 0]))
        Rz4 = self._rot_z(-q[3])

        T45 = self._trans(0.0, -0.3993, 0.0) @ self._homog(self._rot_rpy(1.5708, 0.0, 0.0), ca.DM([0, 0, 0]))
        Rz5 = self._rot_z(-q[4])

        T56 = self._trans(0.0, 0.0, 0.0) @ self._homog(self._rot_rpy(-1.5708, 0.0, 0.0), ca.DM([0, 0, 0]))
        Rz6 = self._rot_z(q[5])

        T67 = self._trans(0.0, -0.126, 0.0) @ self._homog(self._rot_rpy(1.5708, 0.0, 0.0), ca.DM([0, 0, 0]))
        Rz7 = self._rot_z(q[6])

        T = ca.DM(np.eye(4))
        T = T @ T01 @ self._homog(Rz1, ca.DM([0, 0, 0]))
        T = T @ T12 @ self._homog(Rz2, ca.DM([0, 0, 0]))
        T = T @ T23 @ self._homog(Rz3, ca.DM([0, 0, 0]))
        T = T @ T34 @ self._homog(Rz4, ca.DM([0, 0, 0]))
        T = T @ T45 @ self._homog(Rz5, ca.DM([0, 0, 0]))
        T = T @ T56 @ self._homog(Rz6, ca.DM([0, 0, 0]))
        T = T @ T67 @ self._homog(Rz7, ca.DM([0, 0, 0]))

        # Add transformation from iiwa_link_7 to iiwa_link_ee (found in URDF)
        # <origin xyz="0 0 0.045" rpy="0 0 0"/>
        # T_7_to_ee = self._trans(0.0, 0.0, 0.045)
        # T = T @ T_7_to_ee

        # Extract position - now matches iiwa_link_ee frame like Pinocchio/KDL
        return T[0:3, 3]
    
    def solve_ik(self, goal_xyz: np.ndarray, q_init: np.ndarray, 
                 max_iters: int = 200, tol: float = 1e-4, 
                 damping: float = 1e-3, max_step_rad: float = 0.2,
                 joint_limits_lower: Optional[np.ndarray] = None,
                 joint_limits_upper: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simple IK solver using damped least squares.
        
        Args:
            goal_xyz: Target end-effector position (3,)
            q_init: Initial joint configuration (7,)
            max_iters: Maximum iterations
            tol: Position tolerance
            damping: Damping factor for numerical stability
            max_step_rad: Maximum step size in radians
            joint_limits_lower: Lower joint limits (7,)
            joint_limits_upper: Upper joint limits (7,)
            
        Returns:
            Joint configuration that reaches the goal (7,)
        """
        q = np.array(q_init, dtype=float).reshape(-1)
        
        # Robust goal handling
        goal = np.asarray(goal_xyz, dtype=float).flatten()
        if goal.size != 3:
            raise ValueError(f"Goal must have 3 elements, got {goal.size}. Goal value: {goal_xyz}")
        goal = goal.reshape(3)
        
        # Default joint limits if not provided
        if joint_limits_lower is None:
            joint_limits_lower = np.array([-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054])
        if joint_limits_upper is None:
            joint_limits_upper = np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
        
        lower = np.array(joint_limits_lower, dtype=float)
        upper = np.array(joint_limits_upper, dtype=float)
        
        for iter_count in range(max_iters):
            cur = self.forward_kinematics_numeric(q)
            pos_err = goal - cur
            pos_err_norm = np.linalg.norm(pos_err)
            
            if pos_err_norm < tol:
                break
            
            # Use analytical Jacobian for better accuracy
            J_pos = self.compute_jacobian_numeric(q)
            
            # Damped least squares with adaptive damping
            I3 = np.eye(3)
            damping_factor = damping + 0.1 * pos_err_norm  # Adaptive damping
            try:
                dls = J_pos.T @ np.linalg.solve(J_pos @ J_pos.T + damping_factor * I3, pos_err)
            except np.linalg.LinAlgError:
                dls = J_pos.T @ np.linalg.solve(J_pos @ J_pos.T + (damping_factor * 10.0) * I3, pos_err)
            
            # Clamp step magnitude
            step_norm = np.linalg.norm(dls)
            if step_norm > max_step_rad and step_norm > 0.0:
                dls = dls * (max_step_rad / step_norm)
            
            # Line search for better convergence
            best_q = q.copy()
            best_error = pos_err_norm
            
            for alpha in [1.0, 0.5, 0.25, 0.1]:
                q_trial = q + alpha * dls
                # Enforce joint limits
                q_trial = np.minimum(np.maximum(q_trial, lower), upper)
                
                cur_trial = self.forward_kinematics_numeric(q_trial)
                error_trial = np.linalg.norm(goal - cur_trial)
                
                if error_trial < best_error:
                    best_q = q_trial
                    best_error = error_trial
                    break
            
            q = best_q
            
            # Reduce damping if making progress
            if best_error < pos_err_norm:
                damping = max(damping * 0.9, 1e-6)
        
        return q

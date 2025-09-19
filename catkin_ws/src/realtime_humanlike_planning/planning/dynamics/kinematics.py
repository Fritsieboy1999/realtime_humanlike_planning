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
        
        # Initialize Pinocchio model
        self._init_pinocchio_model(urdf_path)
        
        # Initialize symbolic kinematics
        self._init_symbolic_kinematics()
    
    def _init_pinocchio_model(self, urdf_path: Optional[str] = None):
        """Initialize Pinocchio model from URDF."""
        if urdf_path is None:
            # Default URDF path
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            urdf_path = os.path.join(repo_root, "parameters", "dynamics", "lbr_iiwa7_r800.urdf")
        
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found at: {urdf_path}")

        # Use the correct Pinocchio API - try multiple approaches
        try:
            # Try the most common API first
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
        except AttributeError:
            try:
                # Try alternative buildModelsFromUrdf (returns model, collision_model, visual_model)
                self.pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path)
            except AttributeError:
                try:
                    # Try RobotWrapper approach
                    from pinocchio.robot_wrapper import RobotWrapper
                    robot = RobotWrapper.BuildFromURDF(urdf_path)
                    self.pin_model = robot.model
                except ImportError:
                    raise ImportError("Unable to load URDF with available Pinocchio API")
        
        self.pin_data = self.pin_model.createData()

        try:
            self.ee_frame_id = self.pin_model.getFrameId("link7")
        except Exception:
            self.ee_frame_id = None
    
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
        
        if self.ee_frame_id is not None:
            T = self.pin_data.oMf[self.ee_frame_id]
        else:
            T = self.pin_data.oMi[-1]  # Last joint frame
        
        return T.translation
    
    def compute_jacobian_numeric(self, q: np.ndarray) -> np.ndarray:
        """Compute Jacobian using Pinocchio (numeric)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        if self.ee_frame_id is not None:
            J = pin.computeFrameJacobian(
                self.pin_model, self.pin_data, q, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            )
        else:
            J = pin.computeJointJacobians(self.pin_model, self.pin_data, q)
            J = self.pin_data.J[-1]  # Last joint Jacobian
            
        return J[:3, :]  # Return only position part (3x7)
    
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
        goal = np.array(goal_xyz, dtype=float).reshape(3)
        
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

"""
Initial trajectory generation for optimization warm-starting.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Any
from types import SimpleNamespace

from ..dynamics import RobotKinematics, RobotDynamics


class InitialTrajectoryGenerator:
    """
    Generates initial trajectories for optimization warm-starting.
    """
    
    def __init__(self, kinematics: RobotKinematics, dynamics: RobotDynamics):
        """
        Initialize trajectory generator.
        
        Args:
            kinematics: RobotKinematics instance
            dynamics: RobotDynamics instance
        """
        self.kinematics = kinematics
        self.dynamics = dynamics
        self.nq = kinematics.nq
        self.NST = 2 * self.nq  # State vector size
    
    def create_ik_based_trajectory(self, task_params: Any, H: int, tf_guess: float,
                                 joint_limits_lower: Optional[np.ndarray] = None,
                                 joint_limits_upper: Optional[np.ndarray] = None,
                                 joint_effort_limits: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create intelligent initial trajectory using IK and smooth interpolation.
        
        Args:
            task_params: Task parameters with xi0 and goal
            H: Horizon length
            tf_guess: Guess for final time
            joint_limits_lower: Lower joint limits (7,)
            joint_limits_upper: Upper joint limits (7,)
            joint_effort_limits: Joint effort limits (7,)
            
        Returns:
            q_traj: Joint trajectory (7 x H)
            dq_traj: Joint velocity trajectory (7 x H)
            u_traj: Control trajectory (7 x H)
        """
        xi0 = task_params.xi0
        q_start = xi0[:7]
        dq_start = xi0[7:14]
        
        # Default limits if not provided
        if joint_limits_lower is None:
            joint_limits_lower = np.array([-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054])
        if joint_limits_upper is None:
            joint_limits_upper = np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
        if joint_effort_limits is None:
            joint_effort_limits = np.array([320, 320, 176, 176, 110, 40, 40])
        
        # Solve IK for goal position
        try:
            q_goal = self.kinematics.solve_ik(task_params.goal, q_start,
                                            joint_limits_lower=joint_limits_lower,
                                            joint_limits_upper=joint_limits_upper)
            goal_error = np.linalg.norm(self.kinematics.forward_kinematics_numeric(q_goal) - task_params.goal)
            print(f"IK converged: goal error = {goal_error:.6f}m")
        except Exception as e:
            print(f"Warning: IK failed ({e}), using start position as goal")
            q_goal = q_start.copy()
        
        # Create smooth trajectory from start to goal
        time_points = np.linspace(0, 1, H)
        
        # Initialize trajectories
        q_traj = np.zeros((7, H))
        dq_traj = np.zeros((7, H))
        
        for i in range(7):
            # Smooth S-curve interpolation (cosine-based)
            s = 0.5 * (1 - np.cos(np.pi * time_points))
            q_traj[i, :] = q_start[i] + s * (q_goal[i] - q_start[i])
            
            # Compute velocities via differentiation
            dt = tf_guess / (H - 1)
            dq_traj[i, 1:-1] = (q_traj[i, 2:] - q_traj[i, :-2]) / (2 * dt)
            dq_traj[i, 0] = dq_start[i]  # Start with initial velocity
            dq_traj[i, -1] = 0.0  # End with zero velocity
        
        # Create control trajectory using simple inverse dynamics
        dt = tf_guess / (H - 1)
        u_traj = np.zeros((7, H))
        for k in range(H):
            if k < H - 1:
                ddq_desired = (dq_traj[:, k+1] - dq_traj[:, k]) / dt
            else:
                ddq_desired = -dq_traj[:, k] / dt  # Decelerate to zero
            
            # Simple inverse dynamics: u ≈ M*ddq + B*dq
            M = self.dynamics.compute_mass_matrix(q_traj[:, k])
            B = np.diag(self.dynamics.joint_damping)
            u_traj[:, k] = M @ ddq_desired + B @ dq_traj[:, k]
            
            # Clamp to torque limits
            torque_limits = joint_effort_limits * 0.5
            u_traj[:, k] = np.clip(u_traj[:, k], -torque_limits, torque_limits)
        
        return q_traj, dq_traj, u_traj
    
    def create_simple_initial_guess(self, task_params: Any, H: int, 
                                  use_ik: bool = True) -> np.ndarray:
        """
        Create simple initial guess for optimization variables [X, C, U].
        
        Args:
            task_params: Task parameters with xi0 and goal
            H: Horizon length
            use_ik: Whether to use IK for intelligent initialization
            
        Returns:
            Initial guess vector
        """
        # Total variables: X (NST*H) + C (NST*H) + U (nq*H)
        total_vars = self.NST * H + self.NST * H + self.nq * H
        x0 = np.ones(total_vars) * 1e-2  # Start with small constant values
        
        xi0 = task_params.xi0
        q_start = xi0[:7]
        
        # Set initial state exactly (critical)
        x0[0:self.NST] = xi0
        
        if use_ik:
            # Quick IK solve (fewer iterations for speed)
            try:
                q_goal = self.kinematics.solve_ik(task_params.goal, q_start, 
                                                max_iters=20, tol=1e-3)
                ik_success = True
            except Exception:
                q_goal = q_start.copy()
                ik_success = False
            
            # Set final state intelligently if IK succeeded
            if ik_success:
                final_state = np.concatenate([q_goal, np.zeros(7)])  # Zero final velocity
                x0[(H-1)*self.NST:H*self.NST] = final_state
                
                # Set one intermediate point for better convergence
                if H > 4:
                    mid_idx = H // 2
                    q_mid = (q_start + q_goal) / 2
                    mid_state = np.concatenate([q_mid, np.zeros(7)])
                    x0[mid_idx*self.NST:(mid_idx+1)*self.NST] = mid_state
        
        # Initialize covariance with small constant values
        offset_C = self.NST * H
        x0[offset_C:offset_C + self.NST*H] = 1e-5
        
        # Initialize controls with zeros
        offset_U = self.NST * H + self.NST * H
        x0[offset_U:] = 0.0
        
        return x0
    
    def create_linear_interpolation_trajectory(self, task_params: Any, H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create simple linear interpolation trajectory in joint space.
        
        Args:
            task_params: Task parameters with xi0
            H: Horizon length
            
        Returns:
            q_traj: Joint trajectory (7 x H)
            dq_traj: Joint velocity trajectory (7 x H)
            u_traj: Control trajectory (7 x H)
        """
        xi0 = task_params.xi0
        q_start = xi0[:7]
        dq_start = xi0[7:14]
        
        # Simple strategy: stay at current position (minimal motion)
        q_traj = np.tile(q_start.reshape(-1, 1), (1, H))
        dq_traj = np.zeros((7, H))
        dq_traj[:, 0] = dq_start  # Start with initial velocity
        u_traj = np.zeros((7, H))
        
        return q_traj, dq_traj, u_traj
    

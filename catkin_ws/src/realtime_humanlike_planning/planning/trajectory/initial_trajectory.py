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
    
    def create_warm_start_from_previous(self, previous_solution: np.ndarray, 
                                      task_params: Any, H: int) -> Optional[np.ndarray]:
        """
        Create warm start by intelligently adapting previous solution.
        
        This method performs several adaptations:
        1. Adjusts initial state to match new task
        2. Adapts final state using IK for new goal
        3. Smoothly interpolates intermediate states
        4. Scales covariances appropriately
        5. Adjusts control inputs for new trajectory
        
        Args:
            previous_solution: Previous optimization solution
            task_params: Current task parameters
            H: Current horizon length
            
        Returns:
            Adapted solution or None if adaptation fails
        """
        try:
            # Check if previous solution has compatible size
            expected_size = self.NST * H + self.NST * H + self.nq * H
            if previous_solution.shape[0] != expected_size:
                print(f"Warning: Previous solution size mismatch ({previous_solution.shape[0]} vs {expected_size})")
                return None
            
            # Extract previous trajectories
            nX = self.NST * H
            nC = self.NST * H
            
            X_prev = previous_solution[0:nX].reshape(self.NST, H, order='F')
            C_prev = previous_solution[nX:nX+nC].reshape(self.NST, H, order='F')
            U_prev = previous_solution[nX+nC:].reshape(self.nq, H, order='F')
            
            # Adapt state trajectory intelligently
            X_adapted = self._adapt_state_trajectory(X_prev, task_params, H)
            
            # Adapt covariance trajectory
            C_adapted = self._adapt_covariance_trajectory(C_prev, X_prev, X_adapted, H)
            
            # Adapt control trajectory
            U_adapted = self._adapt_control_trajectory(U_prev, X_prev, X_adapted, H)
            
            # Rebuild solution vector
            x0_adapted = np.concatenate([
                X_adapted.ravel(order='F'),
                C_adapted.ravel(order='F'),
                U_adapted.ravel(order='F')
            ])
            
            print(f"Successfully adapted previous solution for warm start")
            return x0_adapted
            
        except Exception as e:
            print(f"Warning: Warm start adaptation failed: {e}")
            return None
    
    def _adapt_state_trajectory(self, X_prev: np.ndarray, task_params: Any, H: int) -> np.ndarray:
        """Adapt state trajectory for new task."""
        X_adapted = X_prev.copy()
        
        # Set new initial state
        X_adapted[:, 0] = task_params.xi0
        
        # Try to solve IK for new goal
        q_prev_final = X_prev[:7, -1]
        try:
            q_new_goal = self.kinematics.solve_ik(task_params.goal, q_prev_final, 
                                                max_iters=20, tol=1e-4)
            X_adapted[:7, -1] = q_new_goal
            X_adapted[7:14, -1] = 0.0  # Zero final velocity
            ik_success = True
        except Exception:
            # If IK fails, use previous final configuration
            ik_success = False
        
        # Smooth interpolation between new initial and final states
        if ik_success:
            q_start = task_params.xi0[:7]
            q_goal = X_adapted[:7, -1]
            
            # Use smooth S-curve interpolation for better trajectories
            time_points = np.linspace(0, 1, H)
            s = 0.5 * (1 - np.cos(np.pi * time_points))
            
            for i in range(7):
                X_adapted[i, :] = q_start[i] + s * (q_goal[i] - q_start[i])
            
            # Recompute velocities
            for k in range(1, H-1):
                X_adapted[7:14, k] = (X_adapted[:7, k+1] - X_adapted[:7, k-1]) / (2 * 0.1)  # Assume dt=0.1
            X_adapted[7:14, 0] = task_params.xi0[7:14]  # Initial velocity
            X_adapted[7:14, -1] = 0.0  # Final velocity
        
        return X_adapted
    
    def _adapt_covariance_trajectory(self, C_prev: np.ndarray, X_prev: np.ndarray, 
                                   X_adapted: np.ndarray, H: int) -> np.ndarray:
        """Adapt covariance trajectory based on state changes."""
        C_adapted = C_prev.copy()
        
        # Scale covariances based on how much the trajectory changed
        for k in range(H):
            state_change = np.linalg.norm(X_adapted[:, k] - X_prev[:, k])
            if state_change > 0.1:  # Significant change
                # Increase covariance to account for uncertainty
                scale_factor = 1.0 + 0.5 * state_change
                C_adapted[:, k] = C_prev[:, k] * scale_factor
            
            # Ensure minimum covariance
            C_adapted[:, k] = np.maximum(C_adapted[:, k], 1e-6)
        
        return C_adapted
    
    def _adapt_control_trajectory(self, U_prev: np.ndarray, X_prev: np.ndarray,
                                X_adapted: np.ndarray, H: int) -> np.ndarray:
        """Adapt control trajectory for new state trajectory."""
        U_adapted = U_prev.copy()
        
        # Recompute controls using simple inverse dynamics where trajectory changed significantly
        for k in range(H-1):
            state_change = np.linalg.norm(X_adapted[:, k] - X_prev[:, k])
            
            if state_change > 0.05:  # Significant change, recompute control
                q_k = X_adapted[:7, k]
                dq_k = X_adapted[7:14, k]
                
                # Desired acceleration
                if k < H-1:
                    dq_next = X_adapted[7:14, k+1]
                    ddq_desired = (dq_next - dq_k) / 0.1  # Assume dt=0.1
                else:
                    ddq_desired = -dq_k / 0.1  # Decelerate to zero
                
                # Simple inverse dynamics
                try:
                    M = self.dynamics.compute_mass_matrix(q_k)
                    B = np.diag(self.dynamics.joint_damping)
                    U_adapted[:, k] = M @ ddq_desired + B @ dq_k
                    
                    # Clamp to reasonable limits
                    torque_limits = np.array([160, 160, 88, 88, 55, 20, 20])  # Conservative limits
                    U_adapted[:, k] = np.clip(U_adapted[:, k], -torque_limits, torque_limits)
                except Exception:
                    # Keep previous control if dynamics computation fails
                    pass
        
        return U_adapted

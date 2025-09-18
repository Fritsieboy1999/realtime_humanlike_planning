"""
Cost function implementations for the optimization problem.
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from typing import Dict, List, Union, Optional, Callable

from .scaling import TemporalScaling


class CostFunctions:
    """
    Collection of cost functions for the optimization problem.
    """
    
    def __init__(self, 
                 fwd_kin_fn: Callable,
                 jac_fn: Callable,
                 H: int,
                 nq: int = 7,
                 NST: int = 14,
                 NGOAL: int = 3):
        """
        Initialize cost functions.
        
        Args:
            fwd_kin_fn: Forward kinematics function
            jac_fn: Jacobian function
            H: Horizon length
            nq: Number of joints
            NST: State vector size
            NGOAL: Goal space dimension
        """
        self.fwd_kin_fn = fwd_kin_fn
        self.jac_fn = jac_fn
        self.H = H
        self.nq = nq
        self.NST = NST
        self.NGOAL = NGOAL
        self.scaling = TemporalScaling()
    
    def input_adapted_cost(self, X: ca.MX, C: ca.MX, U: ca.MX, 
                          task_width: Union[ca.MX, float],
                          weight: float) -> ca.MX:
        """
        Adapted input change cost scaled by endpoint covariance and target width.
        
        Args:
            X: State trajectory (NST x H)
            C: Covariance trajectory (NST x H)
            U: Control trajectory (nq x H)
            task_width: Task width parameter
            weight: Cost weight
            
        Returns:
            Cost contribution
        """
        cost = 0
        
        # Need at least 2 control inputs to compute change of input
        if U.size2() > 1:
            for k in range(self.H-2):  # Control input changes are H-2 in length (for H-1 control inputs)
                # Get joint positions and covariances at time k
                q_k = X[:self.nq, k]
                c_k = C[:self.nq, k]  # Position covariances (diagonal)
                
                # Compute Jacobian for end-effector using CasADi function
                J_k = self.jac_fn(q_k)  # 3 x nq Jacobian
                
                # End-effector covariance: J * diag(c_k) * J^T
                C_diag = ca.diag(c_k)
                Sig_x = J_k @ C_diag @ J_k.T  # 3x3 end-effector covariance
                
                # Symmetric dip scaling approach
                t_norm = k / float(self.H - 1) if self.H > 1 else 0.0
                
                # Calculate d_scale dynamically based on task width
                d_scale, A_scale = self.scaling.compute_dynamic_scaling_parameters(task_width)
                
                # Apply symmetric area-normalized scaling s(t)
                s = self.scaling.symmetric_dip_scaling_casadi(t_norm, A_scale, d_scale)
                
                # Change in control input from time k to k+1
                u_diff_k = U[:, k+1] - U[:, k]
                cost += weight * s * ca.sumsqr(u_diff_k)
        
        return cost
    
    def input_cost(self, U: ca.MX, weight: float) -> ca.MX:
        """
        Input penalty (control effort).
        
        Args:
            U: Control trajectory (nq x H)
            weight: Cost weight
            
        Returns:
            Cost contribution
        """
        return weight * ca.sumsqr(U)
    
    def distance_cost(self, X: ca.MX, goal: ca.MX, weight: float) -> ca.MX:
        """
        End-effector distance penalty.
        
        Args:
            X: State trajectory (NST x H)
            goal: Goal position (3,)
            weight: Cost weight
            
        Returns:
            Cost contribution
        """
        cost = 0
        for k in range(self.H):
            q_k = X[:self.nq, k]
            ee_k = self.fwd_kin_fn(q_k)
            dist_squared = ca.sumsqr(ee_k - goal)
            cost += weight * dist_squared
        return cost
    
    def path_straightness_cost(self, X: ca.MX, xi0: ca.MX, goal: ca.MX, weight: float) -> ca.MX:
        """
        Path straightness penalty - penalize deviations from straight line.
        
        Args:
            X: State trajectory (NST x H)
            xi0: Initial state
            goal: Goal position (3,)
            weight: Cost weight
            
        Returns:
            Cost contribution
        """
        # Start end-effector position from the parameter xi0
        q0_par = xi0[:self.nq]
        p0 = self.fwd_kin_fn(q0_par)  # 3x1
        
        # Direction vector from start to goal (unit)
        v = goal - p0
        v_norm = ca.sqrt(ca.sumsqr(v)) + 1e-12
        v_hat = v / v_norm
        
        # Projection matrix onto the plane perpendicular to v_hat
        I3 = ca.DM.eye(3)
        P_perp = I3 - ca.mtimes(v_hat, v_hat.T)  # 3x3
        
        # Penalize perpendicular distance to the line
        cost = 0
        for k in range(self.H):
            q_k = X[:self.nq, k]
            ee_k = self.fwd_kin_fn(q_k)  # 3x1
            d_perp_k = ca.mtimes(P_perp, (ee_k - p0))  # lateral deviation
            cost += weight * ca.sumsqr(d_perp_k)
        
        return cost
    
    def torque_derivative_cost(self, U: ca.MX, weight: float) -> ca.MX:
        """
        Torque derivative penalty (change in torque).
        
        Args:
            U: Control trajectory (nq x H)
            weight: Cost weight
            
        Returns:
            Cost contribution
        """
        cost = 0
        # For H-1 control inputs, we have H-2 torque derivatives
        if U.size2() > 1:  # Need at least 2 control inputs to compute derivatives
            U_diff = U[:, 1:] - U[:, :-1]  # Differences between consecutive control inputs
            cost = weight * ca.sumsqr(U_diff)
        return cost
    
    def end_effector_covariance_cost(self, X: ca.MX, C: ca.MX, weight: float) -> ca.MX:
        """
        End effector covariance penalty - penalize uncertainty in end effector position.
        
        Args:
            X: State trajectory (NST x H)
            C: Covariance trajectory (NST x H)
            weight: Cost weight
            
        Returns:
            Cost contribution
        """
        cost = 0
        for k in range(self.H):
            q_k = X[:self.nq, k]
            c_k = C[:self.nq, k]  # position covariances (diag in joint space)
            J_k = self.jac_fn(q_k)  # 3x7
            
            # End-effector covariance: J * diag(c_k) * J^T
            Sig_x = ca.mtimes([J_k, ca.diag(c_k), J_k.T])
            
            # Penalize the trace of the end-effector covariance matrix (total variance)
            cost += weight * ca.trace(Sig_x)
        
        return cost
    
    def endpoint_jerk_cost(self, X: ca.MX, h: float, weight: float) -> ca.MX:
        """
        Endpoint jerk penalty - penalize third derivative of end-effector position.
        
        Args:
            X: State trajectory (NST x H)
            h: Time step
            weight: Cost weight
            
        Returns:
            Cost contribution
        """
        cost = 0
        
        # Compute end-effector positions for all time steps
        ee_positions = []
        for k in range(self.H):
            q_k = X[:self.nq, k]
            ee_k = self.fwd_kin_fn(q_k)
            ee_positions.append(ee_k)
        
        # Compute jerk as third-order finite difference of end-effector positions
        # For H time steps, we can compute H-3 jerk values
        if self.H >= 4:  # Need at least 4 points to compute jerk
            for k in range(self.H - 3):
                # Third-order finite difference: jerk = (p[k+3] - 3*p[k+2] + 3*p[k+1] - p[k]) / h^3
                p_k = ee_positions[k]
                p_k1 = ee_positions[k + 1] 
                p_k2 = ee_positions[k + 2]
                p_k3 = ee_positions[k + 3]
                
                jerk_k = (p_k3 - 3*p_k2 + 3*p_k1 - p_k) / (h**3)
                cost += weight * ca.sumsqr(jerk_k)
        
        return cost
    
    def terminal_covariance_cost(self, X: ca.MX, C: ca.MX, task_width: Union[ca.MX, float], 
                               rescale: float, weight: float = 100) -> ca.MX:
        """
        Terminal cost: Final endpoint covariance penalty.
        
        Args:
            X: State trajectory (NST x H)
            C: Covariance trajectory (NST x H)
            task_width: Task width parameter
            rescale: Rescaling factor
            weight: Cost weight
            
        Returns:
            Cost contribution
        """
        x_final = X[:, -1]
        c_final = C[:, -1]
        q_final = x_final[:self.nq]
        cdiag_final = c_final[:self.nq]
        
        # Use accurate URDF Jacobian
        J_final = self.jac_fn(q_final)
        VarEP_final = ca.mtimes([J_final, ca.diag(cdiag_final), J_final.T])
        terminal_cost = weight * task_width * rescale * ca.trace(VarEP_final)
        
        return terminal_cost
    
    def compute_total_cost(self, X: ca.MX, C: ca.MX, U: ca.MX, 
                          xi0: ca.MX, goal: ca.MX, task_width: Union[ca.MX, float],
                          cost_types: List[str], cost_params: Dict[str, float],
                          h: float) -> ca.MX:
        """
        Compute total cost based on enabled cost types.
        
        Args:
            X: State trajectory (NST x H)
            C: Covariance trajectory (NST x H)
            U: Control trajectory (nq x H)
            xi0: Initial state
            goal: Goal position (3,)
            task_width: Task width parameter
            cost_types: List of enabled cost types
            cost_params: Dictionary of cost parameters
            h: Time step
            
        Returns:
            Total cost
        """
        total_cost = 0
        
        for cost_type in cost_types:
            if cost_type == "input":
                total_cost += self.input_cost(U, cost_params.get("ctrl_cost", 1.0))
            elif cost_type == "input_adapted":
                total_cost += self.input_adapted_cost(X, C, U, task_width, 
                                                    cost_params.get("input_adapted_cost", 1.0))
            elif cost_type == "distance":
                total_cost += self.distance_cost(X, goal, cost_params.get("distance_cost", 1.0))
            elif cost_type == "path_straightness":
                total_cost += self.path_straightness_cost(X, xi0, goal, 
                                                        cost_params.get("path_straightness_cost", 1.0))
            elif cost_type == "torque_derivative":
                total_cost += self.torque_derivative_cost(U, cost_params.get("torque_derivative_cost", 1.0))
            elif cost_type == "end_effector_covariance":
                total_cost += self.end_effector_covariance_cost(X, C, 
                                                              cost_params.get("end_effector_covariance_cost", 1.0))
            elif cost_type == "endpoint_jerk":
                total_cost += self.endpoint_jerk_cost(X, h, cost_params.get("endpoint_jerk_cost", 1.0))
            else:
                raise ValueError(f"Unknown cost_type: {cost_type}")
        
        # Add terminal cost if enabled
        if cost_params.get("use_terminal_cost", False):
            total_cost += self.terminal_covariance_cost(X, C, task_width, 
                                                      cost_params.get("rescale", 1.0))
        
        return total_cost

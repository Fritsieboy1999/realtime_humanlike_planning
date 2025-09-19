"""
Post-solve cost analysis for plotting and evaluation.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any
from types import SimpleNamespace

from .scaling import TemporalScaling


class PostSolveCostAnalysis:
    """
    Analyzes costs after optimization for plotting and evaluation.
    """
    
    def __init__(self, fwd_kin_fn, jac_fn, H: int, nq: int = 7):
        """
        Initialize post-solve cost analysis.
        
        Args:
            fwd_kin_fn: Forward kinematics function
            jac_fn: Jacobian function
            H: Horizon length
            nq: Number of joints
        """
        self.fwd_kin_fn = fwd_kin_fn
        self.jac_fn = jac_fn
        self.H = H
        self.nq = nq
        self.scaling = TemporalScaling()
    
    def analyze_costs(self, result: SimpleNamespace, task_params: Any, 
                     cost_types: List[str], cost_params: Dict[str, float],
                     h: float, width: float) -> Dict[str, Any]:
        """
        Analyze all cost components after optimization.
        
        Args:
            result: Optimization result with mu, cov, tau
            task_params: Task parameters
            cost_types: List of enabled cost types
            cost_params: Dictionary of cost parameters
            h: Time step
            width: Task width
            
        Returns:
            Dictionary containing cost breakdown and analysis
        """
        Xo = result.mu  # State trajectory
        Co = result.cov  # Covariance trajectory
        Uo = result.tau  # Control trajectory
        xi0 = task_params.xi0
        goal = task_params.goal
        
        cost_breakdown = {}
        total_running_cost = 0
        
        # Terminal cost
        if cost_params.get("use_terminal_cost", False):
            terminal_cost = self._compute_terminal_cost(Xo, Co, width, cost_params.get("rescale", 1.0))
            cost_breakdown["terminal_cov"] = terminal_cost
        else:
            cost_breakdown["terminal_cov"] = 0
        
        # Process each cost type
        for cost_type in cost_types:
            if cost_type == "input":
                input_cost = cost_params.get("ctrl_cost", 1.0) * np.sum(Uo**2)
                cost_breakdown["input_cost"] = input_cost
                total_running_cost += input_cost
                
            elif cost_type == "input_adapted":
                input_adapted_cost, temporal_scaling_data = self._compute_input_adapted_cost(
                    Xo, Co, Uo, width, cost_params.get("input_adapted_cost", 1.0), h)
                cost_breakdown["input_adapted_cost"] = input_adapted_cost
                cost_breakdown["temporal_scaling"] = temporal_scaling_data
                total_running_cost += input_adapted_cost
                
            elif cost_type == "distance":
                distance_cost = self._compute_distance_cost(Xo, goal, cost_params.get("distance_cost", 1.0))
                cost_breakdown["distance_cost"] = distance_cost
                total_running_cost += distance_cost
                
            elif cost_type == "path_straightness":
                path_cost = self._compute_path_straightness_cost(Xo, xi0, goal, 
                                                               cost_params.get("path_straightness_cost", 1.0))
                cost_breakdown["path_straightness_cost"] = path_cost
                total_running_cost += path_cost
                
            elif cost_type == "torque_derivative":
                torque_deriv_cost = self._compute_torque_derivative_cost(Uo, 
                                                                       cost_params.get("torque_derivative_cost", 1.0))
                cost_breakdown["torque_derivative_cost"] = torque_deriv_cost
                total_running_cost += torque_deriv_cost
                
            elif cost_type == "end_effector_covariance":
                ee_cov_cost = self._compute_end_effector_covariance_cost(Xo, Co, 
                                                                       cost_params.get("end_effector_covariance_cost", 1.0))
                cost_breakdown["end_effector_covariance_cost"] = ee_cov_cost
                total_running_cost += ee_cov_cost
                
            elif cost_type == "endpoint_jerk":
                jerk_cost = self._compute_endpoint_jerk_cost(Xo, h, cost_params.get("endpoint_jerk_cost", 1.0))
                cost_breakdown["endpoint_jerk_cost"] = jerk_cost
                total_running_cost += jerk_cost
        
        # Combine all cost terms
        cost_breakdown["total_running"] = total_running_cost
        cost_breakdown["total"] = cost_breakdown["terminal_cov"] + total_running_cost
        
        return cost_breakdown
    
    def _compute_terminal_cost(self, Xo: np.ndarray, Co: np.ndarray, width: float, rescale: float) -> float:
        """Compute terminal covariance cost."""
        x_final = Xo[:, -1]
        c_final = Co[:, -1]
        q_final = x_final[:self.nq]
        cdiag_final = c_final[:self.nq]
        
        # Use symbolic functions for post-solve analysis
        J_final = np.array(self.jac_fn(q_final))
        VarEP_final = J_final @ np.diag(cdiag_final) @ J_final.T
        terminal_cost = rescale * 100 * width * np.trace(VarEP_final)
        
        return terminal_cost
    
    def _compute_input_adapted_cost(self, Xo: np.ndarray, Co: np.ndarray, Uo: np.ndarray, 
                                  width: float, weight: float, h: float) -> tuple[float, Dict[str, Any]]:
        """Compute input adapted cost with temporal scaling analysis."""
        input_adapted_cost = 0
        scaling_values = []
        step_normalized_values = []
        control_input_changes = []
        scaled_costs = []
        time_steps = []
        
        for k in range(self.H-2):  # Control input changes are H-2 in length
            # Get joint positions and covariances at time k
            q_k = Xo[:self.nq, k]
            c_k = Co[:self.nq, k]  # Position covariances (diagonal)
            
            # Compute Jacobian for end-effector
            J_k = np.array(self.jac_fn(q_k))  # 3 x nq Jacobian
            
            # End-effector covariance: J * diag(c_k) * J^T
            Sig_x = J_k @ np.diag(c_k) @ J_k.T  # 3x3 end-effector covariance
            
            # Symmetric dip scaling approach (matching optimization)
            t_norm = k / float(self.H - 1) if self.H > 1 else 0.0
            
            # Calculate d_scale dynamically based on task width
            d_scale, A_scale = self.scaling.compute_dynamic_scaling_parameters(width)
            
            # Apply symmetric area-normalized scaling s(t)
            s = self.scaling.symmetric_dip_scaling_numpy(t_norm, A_scale, d_scale)
            
            # Control input change from time k to k+1 (matching optimization)
            u_diff_k = Uo[:, k+1] - Uo[:, k]
            control_change_magnitude = np.sum(u_diff_k**2)
            
            # Scaled cost contribution
            scaled_cost = weight * s * control_change_magnitude
            input_adapted_cost += scaled_cost
            
            # Store values for analysis
            scaling_values.append(s)
            step_normalized_values.append(t_norm)
            control_input_changes.append(control_change_magnitude)
            scaled_costs.append(scaled_cost)
            time_steps.append(k * h)  # Convert to actual time
        
        # Create temporal scaling analysis data
        temporal_scaling_data = {
            'scaling_values': np.array(scaling_values),
            'step_normalized': np.array(step_normalized_values),
            'control_changes': np.array(control_input_changes),
            'scaled_costs': np.array(scaled_costs),
            'time_steps': np.array(time_steps),
            'task_width': width,
            'total_scaled_cost': input_adapted_cost,
            'd_scale': d_scale,
            'A_scale': A_scale
        }
        
        return input_adapted_cost, temporal_scaling_data
    
    def _compute_distance_cost(self, Xo: np.ndarray, goal: np.ndarray, weight: float) -> float:
        """Compute distance cost."""
        distance_cost = 0
        for k in range(self.H):
            ee_k = np.array(self.fwd_kin_fn(Xo[:self.nq, k])).ravel()
            dist_squared = np.sum((ee_k - goal)**2)
            distance_cost += weight * dist_squared
        return distance_cost
    
    def _compute_path_straightness_cost(self, Xo: np.ndarray, xi0: np.ndarray, 
                                      goal: np.ndarray, weight: float) -> float:
        """Compute path straightness cost."""
        q0 = xi0[:self.nq]
        p0 = np.array(self.fwd_kin_fn(q0)).ravel()  # Start EE position
        
        # Direction vector from start to goal (unit)
        v = goal - p0
        v_norm = np.linalg.norm(v) + 1e-12
        v_hat = v / v_norm
        
        # Projection matrix onto the plane perpendicular to v_hat
        I3 = np.eye(3)
        P_perp = I3 - np.outer(v_hat, v_hat)  # 3x3
        
        # Calculate path straightness cost
        path_straightness_cost = 0
        for k in range(self.H):
            ee_k = np.array(self.fwd_kin_fn(Xo[:self.nq, k])).ravel()
            d_perp_k = P_perp @ (ee_k - p0)  # lateral deviation
            path_straightness_cost += weight * np.sum(d_perp_k**2)
        
        return path_straightness_cost
    
    def _compute_torque_derivative_cost(self, Uo: np.ndarray, weight: float) -> float:
        """Compute torque derivative cost."""
        torque_derivative_cost = 0
        if Uo.shape[1] > 1:  # Need at least 2 control inputs to compute derivatives
            U_diff = Uo[:, 1:] - Uo[:, :-1]  # Differences between consecutive control inputs
            torque_derivative_cost = weight * np.sum(U_diff**2)
        return torque_derivative_cost
    
    def _compute_end_effector_covariance_cost(self, Xo: np.ndarray, Co: np.ndarray, weight: float) -> float:
        """Compute end effector covariance cost."""
        ee_covariance_cost = 0
        for k in range(self.H):
            q_k = Xo[:self.nq, k]
            c_k = Co[:self.nq, k]  # Position covariances only
            
            # Compute Jacobian
            J_k = np.array(self.jac_fn(q_k))
            
            # End-effector covariance: J * diag(c_k) * J^T
            Sig_x = J_k @ np.diag(c_k) @ J_k.T
            
            # Penalize the trace of the end-effector covariance matrix
            ee_covariance_cost += weight * np.trace(Sig_x)
        
        return ee_covariance_cost
    
    def _compute_endpoint_jerk_cost(self, Xo: np.ndarray, h: float, weight: float) -> float:
        """Compute endpoint jerk cost."""
        endpoint_jerk_cost = 0
        
        # Compute end-effector positions for all time steps
        ee_positions = []
        for k in range(self.H):
            q_k = Xo[:self.nq, k]
            ee_k = np.array(self.fwd_kin_fn(q_k)).ravel()
            ee_positions.append(ee_k)
        
        # Compute jerk as third-order finite difference of end-effector positions
        if self.H >= 4:  # Need at least 4 points to compute jerk
            for k in range(self.H - 3):
                # Third-order finite difference: jerk = (p[k+3] - 3*p[k+2] + 3*p[k+1] - p[k]) / h^3
                p_k = ee_positions[k]
                p_k1 = ee_positions[k + 1] 
                p_k2 = ee_positions[k + 2]
                p_k3 = ee_positions[k + 3]
                
                jerk_k = (p_k3 - 3*p_k2 + 3*p_k1 - p_k) / (h**3)
                endpoint_jerk_cost += weight * np.sum(jerk_k**2)
        
        return endpoint_jerk_cost

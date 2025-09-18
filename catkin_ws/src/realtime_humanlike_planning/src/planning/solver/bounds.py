"""
Bounds computation for optimization variables and constraints.
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from typing import Tuple, Optional


class BoundsComputer:
    """
    Computes bounds for optimization variables and constraints.
    """
    
    def __init__(self, 
                 H: int,
                 nq: int = 7,
                 NST: int = 14,
                 NGOAL: int = 3,
                 cov_min: float = 1e-6,
                 joint_limits_lower: Optional[np.ndarray] = None,
                 joint_limits_upper: Optional[np.ndarray] = None,
                 joint_velocity_limits: Optional[np.ndarray] = None,
                 joint_effort_limits: Optional[np.ndarray] = None):
        """
        Initialize bounds computer.
        
        Args:
            H: Horizon length
            nq: Number of joints
            NST: State vector size
            NGOAL: Goal space dimension
            cov_min: Minimum covariance value
            joint_limits_lower: Lower joint limits (7,)
            joint_limits_upper: Upper joint limits (7,)
            joint_velocity_limits: Joint velocity limits (7,)
            joint_effort_limits: Joint effort limits (7,)
        """
        self.H = H
        self.nq = nq
        self.NST = NST
        self.NGOAL = NGOAL
        self.cov_min = cov_min
        
        # Default KUKA LBR iiwa7 limits if not provided
        if joint_limits_lower is None:
            joint_limits_lower = np.array([-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054])
        if joint_limits_upper is None:
            joint_limits_upper = np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
        if joint_velocity_limits is None:
            joint_velocity_limits = np.array([1.483, 1.483, 1.745, 1.308, 2.268, 2.356, 2.356])
        if joint_effort_limits is None:
            joint_effort_limits = np.array([320, 320, 176, 176, 110, 40, 40])
        
        self.joint_limits_lower = joint_limits_lower
        self.joint_limits_upper = joint_limits_upper
        self.joint_velocity_limits = joint_velocity_limits
        self.joint_effort_limits = joint_effort_limits
    
    def compute_variable_bounds(self) -> Tuple[ca.DM, ca.DM]:
        """
        Compute bounds for decision variables [X, C, U].
        
        Returns:
            lbx: Lower bounds for decision variables
            ubx: Upper bounds for decision variables
        """
        # Total number of variables: X (NST*H) + C (NST*H) + U (nq*H)
        nx_tot = self.NST * self.H + self.NST * self.H + self.nq * self.H
        
        lbx = -ca.inf * ca.DM.ones(nx_tot, 1)
        ubx = ca.inf * ca.DM.ones(nx_tot, 1)
        
        # Joint position and velocity bounds
        vel_safety_margin = 1.0
        vel_limits = self.joint_velocity_limits * vel_safety_margin
        
        for k in range(self.H):
            # Joint position bounds from URDF
            for i in range(self.nq):
                lbx[k * self.NST + i] = self.joint_limits_lower[i]
                ubx[k * self.NST + i] = self.joint_limits_upper[i]
            # Joint velocity bounds from URDF (with safety margin)
            for i in range(self.nq):
                lbx[k * self.NST + self.nq + i] = -vel_limits[i]
                ubx[k * self.NST + self.nq + i] = vel_limits[i]

        # Covariance lower bounds
        off_X = self.NST * self.H
        for i in range(self.NST * self.H):
            lbx[off_X + i] = self.cov_min
            ubx[off_X + i] = 1e8

        # Control bounds
        safety_margin = 1.0
        torque_limits = self.joint_effort_limits * safety_margin
        off_U = self.NST * self.H + self.NST * self.H  # After X and C
        for k in range(self.H):
            for i in range(self.nq):
                lbx[off_U + k * self.nq + i] = -torque_limits[i]
                ubx[off_U + k * self.nq + i] = torque_limits[i]
        
        return lbx, ubx
    
    def compute_constraint_bounds(self) -> Tuple[ca.DM, ca.DM]:
        """
        Compute bounds for constraints.
        
        Returns:
            lbg: Lower bounds for constraints
            ubg: Upper bounds for constraints
        """
        # Count constraints
        n_dyn = (self.H-1) * self.NST * 2  # Mean + cov dynamics
        n_ic = self.NST               # Initial condition
        n_ic_cov = self.NST           # Initial covariance
        n_ic_input = self.nq          # Initial control constraint (zero input at start)
        n_final_pos = self.NGOAL      # Final position constraint
        n_final_vel = self.nq         # Final velocity constraint
        n_fitts = 1                   # Fitts' law time constraint (always present, conditionally active)
        n_pos = self.NST * self.H     # Covariance positivity
        
        ng_tot = n_dyn + n_ic + n_ic_cov + n_ic_input + n_final_pos + n_final_vel + n_fitts + n_pos
        
        lbg = ca.DM.zeros(ng_tot, 1)
        ubg = ca.DM.zeros(ng_tot, 1)

        # Initial covariance bounds (equality constraint)
        lbg[n_dyn + n_ic:n_dyn + n_ic + n_ic_cov] = 0
        ubg[n_dyn + n_ic:n_dyn + n_ic + n_ic_cov] = 0

        # Initial control constraint bounds (zero input at start)
        ic_input_start = n_dyn + n_ic + n_ic_cov
        lbg[ic_input_start:ic_input_start + n_ic_input] = 0
        ubg[ic_input_start:ic_input_start + n_ic_input] = 0

        # Final position constraint bounds (exact)
        pos_start = ic_input_start + n_ic_input
        lbg[pos_start:pos_start + n_final_pos] = 0
        ubg[pos_start:pos_start + n_final_pos] = 0

        # Final velocity constraint bounds (velocity = 0)
        vel_start = pos_start + n_final_pos
        lbg[vel_start:vel_start + n_final_vel] = 0
        ubg[vel_start:vel_start + n_final_vel] = 0

        # Fitts' law time constraint bounds (conditionally active)
        fitts_start = vel_start + n_final_vel
        lbg[fitts_start] = 0  # final_time - fitts_duration = 0 when active, 0 = 0 when inactive
        ubg[fitts_start] = 0

        # Covariance positivity bounds
        cov_start = fitts_start + n_fitts
        lbg[cov_start:] = self.cov_min
        ubg[cov_start:] = ca.inf
        
        return lbg, ubg

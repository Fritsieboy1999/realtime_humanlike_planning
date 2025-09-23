"""
Dynamics module for robot mass matrix computation and linearization.
"""

from __future__ import annotations

import numpy as np
import casadi as ca
import pinocchio as pin
from typing import Optional, Tuple

from .kinematics import RobotKinematics


class RobotDynamics:
    """
    Robot dynamics handler with mass matrix computation and linearization.
    """
    
    def __init__(self, kinematics: RobotKinematics, joint_damping: Optional[np.ndarray] = None):
        """
        Initialize dynamics with kinematics and damping parameters.
        
        Args:
            kinematics: RobotKinematics instance
            joint_damping: Joint damping coefficients (7,)
        """
        self.kinematics = kinematics
        self.nq = kinematics.nq
        
        # Default joint damping if not provided
        if joint_damping is None:
            joint_damping = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.joint_damping = joint_damping
        
        # Initialize frozen linearization
        self._init_frozen_linearization()
    
    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute mass matrix using Pinocchio.
        
        Args:
            q: Joint configuration (7,)
            
        Returns:
            Mass matrix (7x7)
        """
        if self.kinematics.pin_model is None:
            # Fallback to identity matrix if Pinocchio is not available
            print("⚠️ Using identity mass matrix (Pinocchio disabled)")
            return np.eye(self.nq)
            
        q_np = np.array(q, dtype=float).reshape(-1)
        
        # Try different Pinocchio API versions
        try:
            pin.computeAllTerms(self.kinematics.pin_model, self.kinematics.pin_data, q_np, np.zeros(self.nq))
        except AttributeError:
            # Try alternative API
            try:
                pin.crba(self.kinematics.pin_model, self.kinematics.pin_data, q_np)
            except AttributeError:
                print("⚠️ Using identity mass matrix (Pinocchio API incompatible)")
                return np.eye(self.nq)
                
        return np.array(self.kinematics.pin_data.M)
    
    def _init_frozen_linearization(self):
        """Pre-compute frozen linearization at a nominal configuration for efficiency."""
        # Use home configuration as nominal point
        q_nom = np.array([0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, 0.0])
        
        # Compute nominal mass matrix and damping
        M_nom = self.compute_mass_matrix(q_nom)
        M_inv_nom = np.linalg.inv(M_nom + 0.2 * np.eye(self.nq))
        D_nom = np.diag(self.joint_damping)
        
        # Store for later use
        self.M_nom = M_nom
        self.M_inv_nom = M_inv_nom
        self.D_nom = D_nom
    
    def get_linearized_dynamics(self, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get pre-computed linearized dynamics matrices A and B.
        
        Args:
            h: Time step
            
        Returns:
            A: State transition matrix (14x14)
            B: Control input matrix (14x7)
        """
        Iq = np.eye(self.nq)
        Oq = np.zeros((self.nq, self.nq))
        
        # Implicit damping, explicit position discretization
        # v_{k+1} = (I + h M^{-1} D)^{-1} (v_k + h M^{-1} u_k)
        # q_{k+1} = q_k + h v_k
        V_update_inv = np.linalg.inv(Iq + h * (self.M_inv_nom @ self.D_nom))
        
        A_frozen = np.block([
            [Iq, h * Iq],
            [Oq, V_update_inv]
        ])
        
        B_frozen = np.block([
            [Oq],
            [h * V_update_inv @ self.M_inv_nom]
        ])
        
        return A_frozen, B_frozen
    
    def get_linearized_dynamics_casadi(self, h: float) -> Tuple[ca.DM, ca.DM]:
        """
        Get pre-computed linearized dynamics matrices A and B as CasADi matrices.
        
        Args:
            h: Time step
            
        Returns:
            A: State transition matrix (14x14) as CasADi DM
            B: Control input matrix (14x7) as CasADi DM
        """
        A_frozen, B_frozen = self.get_linearized_dynamics(h)
        return ca.DM(A_frozen), ca.DM(B_frozen)
    
    def compute_dynamics_at_config(self, q: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute linearized dynamics at a specific configuration.
        
        Args:
            q: Joint configuration (7,)
            h: Time step
            
        Returns:
            A: State transition matrix (14x14)
            B: Control input matrix (14x7)
        """
        # Compute mass matrix at current configuration
        M = self.compute_mass_matrix(q)
        M_inv = np.linalg.inv(M + 0.2 * np.eye(self.nq))
        D = np.diag(self.joint_damping)
        
        Iq = np.eye(self.nq)
        Oq = np.zeros((self.nq, self.nq))
        
        # Build A, B matrices
        V_update_inv = np.linalg.inv(Iq + h * (M_inv @ D))
        
        A = np.block([
            [Iq, h * Iq],
            [Oq, V_update_inv]
        ])
        
        B = np.block([
            [Oq],
            [h * V_update_inv @ M_inv]
        ])
        
        return A, B

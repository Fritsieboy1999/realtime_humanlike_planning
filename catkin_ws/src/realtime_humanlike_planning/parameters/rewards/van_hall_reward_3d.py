"""
Van Hall reward parameters for 3D reaching tasks.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Union


@dataclass
class VanHallRewardParams3D:
    """Reward parameters for Van Hall 3D reaching tasks."""
    
    # Time discretization
    h: float  # Time step (seconds)
    
    # Noise parameters
    sigma_tau: float  # Control-dependent noise
    eps: float  # Signal-independent noise
    
    # Cost weights
    ctrl_cost: float  # Control effort cost
    input_adapted_cost: float  # Adaptive input cost
    torque_derivative_cost: float  # Torque derivative cost
    distance_cost: float  # Distance to goal cost
    path_straightness_cost: float  # Path straightness cost
    end_effector_covariance_cost: float  # End-effector covariance cost
    endpoint_jerk_cost: float  # Endpoint jerk cost
    exp_dist_cost: float  # Exponential distance cost
    
    # Cost type configuration
    cost_type: Union[str, List[str]]  # Type(s) of cost to use
    
    # Scaling and normalization
    rescale: float  # Rescaling factor
    
    # Terminal cost
    use_terminal_cost: bool  # Whether to use terminal cost
    
    # Fitts' law parameters
    use_fitts_law: bool  # Whether to use Fitts' law
    fitts_a: float  # Fitts' law intercept
    fitts_b: float  # Fitts' law slope
    
    # Gaussian reward parameters (if using gaussian_reward cost type)
    width: List[float]  # Task width in each dimension
    gauss_temperature: float  # Temperature for Gaussian reward
    early_width_scale: float  # Early width scaling factor
    discount_factor: float  # Discount factor for temporal weighting
    
    # Temporal scaling parameters
    width_scaling_weight: float  # Weight for width-based scaling
    
    @classmethod
    def default(cls) -> 'VanHallRewardParams3D':
        """Create default Van Hall reward parameters."""
        return cls(
            # Time discretization
            h=0.01,  # 10ms time step
            
            # Noise parameters
            sigma_tau=0.1,
            eps=1e-6,
            
            # Cost weights
            ctrl_cost=1.0,
            input_adapted_cost=1.0,
            torque_derivative_cost=0.1,
            distance_cost=10.0,
            path_straightness_cost=1.0,
            end_effector_covariance_cost=1.0,
            endpoint_jerk_cost=0.01,
            exp_dist_cost=1.0,
            
            # Cost type - use input_adapted as default
            cost_type="input_adapted",
            
            # Scaling and normalization
            rescale=1.0,
            
            # Terminal cost
            use_terminal_cost=True,
            
            # Fitts' law parameters
            use_fitts_law=False,
            fitts_a=0.1,
            fitts_b=0.15,
            
            # Gaussian reward parameters
            width=[0.01, 0.01, 0.01],  # 1cm tolerance in each direction
            gauss_temperature=2.0,
            early_width_scale=3.0,
            discount_factor=0.99,
            
            # Temporal scaling parameters
            width_scaling_weight=1.0,
        )

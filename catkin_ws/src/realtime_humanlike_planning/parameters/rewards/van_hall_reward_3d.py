"""
Van Hall reward parameters for 3D reaching tasks.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, replace, field
from typing import List, Union


@dataclass
class VanHallRewardParams3D:
    """Van Hall 3D reward parameters for human-like reaching with terminal covariance penalty."""
    
    # Terminal cost parameters
    rescale: float = 1000.0                     # scaling factor for terminal cost
    use_terminal_cost: bool = True              # whether to include terminal covariance penalty
    
    # Running cost type and parameters - can be a string or list of strings for combinations
    cost_type: Union[str, List[str]] = field(default_factory=lambda: ["input_adapted", "path_straightness", "endpoint_jerk"])  # "input", "input_adapted", "distance", "path_straightness", "gaussian_reward", "torque_derivative", "end_effector_covariance", "endpoint_jerk", or list for combinations
    ctrl_cost: float = 0.00001                  # control effort penalty (when "input" in cost_type)
    torque_derivative_cost: float = 0.0001      # torque derivative penalty (when "torque_derivative" in cost_type)
    input_adapted_cost: float = 0.0005          # adapted input cost scaling factor (when "input_adapted" in cost_type)
    width_scaling_weight: float = 1.0           # weight for sigmoid width scaling (higher = stronger effect)
    
    distance_cost: float = 10.0                 # EE distance penalty weight (when "distance" in cost_type)
    path_straightness_cost: float = 5.0         # path straightness penalty weight (when "path_straightness" in cost_type)
    endpoint_variance_cost: float = 20          # Gaussian reward weight (when "gaussian_reward" in cost_type) - negative log-likelihood
    end_effector_covariance_cost: float = 1.0   # end effector covariance penalty weight (when "end_effector_covariance" in cost_type)
    endpoint_jerk_cost: float = 1e-6            # endpoint jerk penalty weight (when "endpoint_jerk" in cost_type)
    
    # Dynamics parameters (like in reference)
    h: float = 0.01                             # time step (100 Hz)
    eps: float = 0.0                            # noise parameter
    sigma_tau: float = 0.01                     # control-dependent noise scaling
    
    # Fitts' law parameters: MT = a + b * log2(2D/W)
    fitts_a: float = 0.65                       # Intercept (seconds)
    fitts_b: float = 0.12                       # Slope (seconds per bit)
    use_fitts_law: bool = True                  # Whether to enforce Fitts' law constraint
    
    @staticmethod
    def default():
        return VanHallRewardParams3D()

    def copy_with(self, **kwargs):
        return replace(self, **kwargs)

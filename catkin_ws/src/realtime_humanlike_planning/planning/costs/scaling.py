"""
Temporal scaling functions for cost modulation.
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from scipy.special import beta
from typing import Union


class TemporalScaling:
    """
    Temporal scaling functions for cost modulation over time.
    """
    
    @staticmethod
    def symmetric_dip_scaling_casadi(t_norm: Union[ca.MX, ca.DM, float], 
                                   A: float, 
                                   d: Union[ca.MX, ca.DM, float]) -> Union[ca.MX, ca.DM]:
        """
        Symmetric, area-normalized scaling s(t) in (0,1] using CasADi.
        
        Formula: s(t) = 1 - A * [ t^d * (1 - t)^(1/d) ] / Beta(d+1, 1/d + 1)
        
        Args:
            t_norm: Normalized time in [0,1] (CasADi MX/DM or float)
            A: Area knob (float > 0) e.g., 0.2
            d: Shift knob (CasADi expression or float > 0); d<1 -> earlier minimum, d>1 -> later minimum
            
        Returns:
            s(t_norm) as CasADi expression
        """
        # Clamp t to [eps, 1-eps] for numerical safety
        eps = 1e-12
        t = ca.fmax(eps, ca.fmin(1.0 - eps, t_norm))

        # Kernel k(t) = t^d * (1-t)^(1/d)
        p = d
        q = 1.0 / d  # Assume d > 0 always
        k = ca.power(t, p) * ca.power(1.0 - t, q)

        # Beta function B(p+1, q+1) - handle symbolic d using high-resolution interpolation
        # Pre-compute beta values for range of d values we expect
        d_min, d_max = 0.1, 5.0  # Range covering our expected d_scale values
        d_grid = np.linspace(d_min, d_max, 2000)  # Very high resolution for maximum accuracy
        beta_values = []
        
        for d_val in d_grid:
            try:
                p_val = d_val
                q_val = 1.0 / d_val
                beta_val = beta(p_val + 1.0, q_val + 1.0)
                beta_values.append(beta_val)
            except:
                beta_values.append(1.0)  # Fallback
        
        # Create CasADi interpolant for beta function using cubic splines for smoother interpolation
        beta_interp = ca.interpolant('beta_interp', 'bspline', [d_grid], beta_values)
        
        # Clamp d to valid range and interpolate
        d_clamped = ca.fmax(d_min, ca.fmin(d_max, d))
        B = beta_interp(d_clamped)

        s = 1.0 - A * (k / (B + eps))
        # Keep strictly positive (just in case A is large)
        return ca.fmax(s, 1e-9)
    
    @staticmethod
    def symmetric_dip_scaling_numpy(t_norm: float, A: float, d: float) -> float:
        """
        NumPy version of symmetric, area-normalized scaling s(t) in (0,1] for analysis/plotting.
        Uses exact same formula as CasADi version with proper beta function.
        
        Args:
            t_norm: Normalized time in [0,1]
            A: Area knob (float > 0)
            d: Shift knob (float > 0)
            
        Returns:
            Scaling value s(t_norm)
        """
        # Clamp t to [eps, 1-eps] for numerical safety
        eps = 1e-12
        t = np.clip(t_norm, eps, 1.0 - eps)

        # Kernel k(t) = t^d * (1-t)^(1/d)
        p = d
        q = 1.0 / d if d > 0 else 1.0
        k = np.power(t, p) * np.power(1.0 - t, q)

        # Beta function B(p+1, q+1) - exact calculation
        try:
            B = beta(p + 1.0, q + 1.0)
        except:
            B = 1.0  # Fallback for numerical issues

        s = 1.0 - A * (k / (B + eps))
        # Keep strictly positive (just in case A is large)
        return max(s, 1e-9)
    
    @staticmethod
    def compute_dynamic_scaling_parameters(task_width: Union[float, ca.MX, ca.DM],
                                         min_width: float = 0.005,
                                         max_width: float = 0.03,
                                         left_minimum: float = 0.26,
                                         right_minimum: float = 1.0) -> tuple[Union[float, ca.MX, ca.DM], Union[float, ca.MX, ca.DM]]:
        """
        Compute dynamic scaling parameters based on task width.
        
        Args:
            task_width: Current task width (can be CasADi symbolic)
            min_width: Minimum width for scaling
            max_width: Maximum width for scaling
            left_minimum: d_scale value at min_width
            right_minimum: d_scale value at max_width
            
        Returns:
            d_scale: Computed d_scale parameter
            A_scale: Computed A_scale parameter
        """
        # Linear interpolation for d_scale - handle CasADi variables
        if isinstance(task_width, (ca.MX, ca.DM)):
            width_clamped = ca.fmin(task_width, max_width)
        else:
            width_clamped = min(task_width, max_width)
        
        slope = (right_minimum - left_minimum) / (max_width - min_width)
        d_scale = left_minimum + slope * (width_clamped - min_width)
        
        # Compute A_scale based on d_scale
        A_scale = 0.3 - 0.25 * d_scale
        
        return d_scale, A_scale

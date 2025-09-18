"""
Solver options and configuration for optimization.
"""

from __future__ import annotations

from typing import Dict, Optional


class SolverOptions:
    """
    Solver options configuration for different solver types.
    """
    
    @staticmethod
    def get_mumps_options() -> Dict[str, any]:
        """
        Get optimized MUMPS solver options for smooth human-like motion.
        
        Returns:
            Dictionary of IPOPT options optimized for MUMPS
        """
        return {
            "ipopt.print_level": 0,               # Quiet output
            "ipopt.tol": 1e-3,                   # Good convergence tolerance
            "ipopt.acceptable_tol": 1e-2,        # Reasonable backup tolerance
            "ipopt.acceptable_iter": 3,          # Quick acceptance
            "ipopt.max_iter": 300,               # Reasonable iteration limit
            "ipopt.hessian_approximation": "limited-memory",  # Fast Hessian approx
            "ipopt.linear_solver": "mumps",      # MUMPS for smooth trajectories
            "ipopt.mu_strategy": "monotone",     # Faster barrier updates
            "ipopt.alpha_for_y": "primal",       # Faster dual updates  
            "print_time": 0,
            "ipopt.sb": "yes",
        }
    
    @staticmethod
    def get_ma27_options() -> Dict[str, any]:
        """
        Get MA27 solver options (alternative to MUMPS).
        
        Returns:
            Dictionary of IPOPT options for MA27
        """
        return {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 3,
            "ipopt.max_iter": 300,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.linear_solver": "ma27",
            "ipopt.mu_strategy": "monotone",
            "ipopt.alpha_for_y": "primal",
            "print_time": 0,
            "ipopt.sb": "yes",
        }
    
    @staticmethod
    def get_ma57_options() -> Dict[str, any]:
        """
        Get MA57 solver options (another alternative).
        
        Returns:
            Dictionary of IPOPT options for MA57
        """
        return {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 3,
            "ipopt.max_iter": 300,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.linear_solver": "ma57",
            "ipopt.mu_strategy": "monotone",
            "ipopt.alpha_for_y": "primal",
            "print_time": 0,
            "ipopt.sb": "yes",
        }
    
    @staticmethod
    def get_solver_options(solver_type: str = "mumps", custom_opts: Optional[Dict[str, any]] = None) -> Dict[str, any]:
        """
        Get solver options for specified solver type.
        
        Args:
            solver_type: Type of solver ("mumps", "ma27", "ma57")
            custom_opts: Custom options to override defaults
            
        Returns:
            Dictionary of solver options
        """
        if solver_type.lower() == "mumps":
            opts = SolverOptions.get_mumps_options()
        elif solver_type.lower() == "ma27":
            opts = SolverOptions.get_ma27_options()
        elif solver_type.lower() == "ma57":
            opts = SolverOptions.get_ma57_options()
        else:
            raise ValueError(f"Unknown solver type: {solver_type}. Supported: mumps, ma27, ma57")
        
        # Override with custom options if provided
        if custom_opts:
            opts.update(custom_opts)
        
        return opts

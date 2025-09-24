"""
Default 3D task parameters.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskParams3D:
    """3D task parameters."""
    
    # Initial state [q1...q7, dq1...dq7]
    xi0: np.ndarray
    
    # Goal position in 3D space
    goal: np.ndarray
    
    # Task width/tolerance
    width: float
    
    @classmethod
    def create_reaching_task(cls, 
                           q_start: Optional[np.ndarray] = None,
                           dq_start: Optional[np.ndarray] = None,
                           goal_position: Optional[np.ndarray] = None,
                           width: float = 0.01) -> 'TaskParams3D':
        """
        Create a reaching task.
        
        Args:
            q_start: Initial joint positions (7,)
            dq_start: Initial joint velocities (7,)
            goal_position: Goal position in 3D (3,)
            width: Task width/tolerance (meters)
            
        Returns:
            TaskParams3D instance
        """
        # Default values
        if q_start is None:
            q_start = np.array([0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, 0.0])
        if dq_start is None:
            dq_start = np.zeros(7)
        if goal_position is None:
            goal_position = np.array([0.5, 0.0, 0.5])  # 50cm forward, 50cm up
        
        xi0 = np.concatenate([q_start, dq_start])
        
        # Ensure goal_position is proper 3D array
        goal_array = np.asarray(goal_position, dtype=float).flatten()
        if goal_array.size != 3:
            raise ValueError(f"Goal position must have 3 elements, got {goal_array.size}")
        
        return cls(
            xi0=xi0,
            goal=goal_array,
            width=width
        )

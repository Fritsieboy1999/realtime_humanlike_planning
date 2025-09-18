"""
Fitts' law implementation for movement time prediction.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any


class FittsLaw:
    """
    Fitts' law implementation for predicting movement time based on task difficulty.
    
    Fitts' law: MT = a + b * log2(2D/W)
    where:
    - MT: Movement time
    - D: Distance to target
    - W: Target width (tolerance)
    - a, b: Empirically determined constants
    """
    
    def __init__(self, a: float = 0.1, b: float = 0.15):
        """
        Initialize Fitts' law with empirical constants.
        
        Args:
            a: Intercept constant (seconds)
            b: Slope constant (seconds)
        """
        self.a = a
        self.b = b
    
    def calculate_movement_time(self, distance: float, width: float) -> float:
        """
        Calculate movement time using Fitts' law.
        
        Args:
            distance: Distance to target (meters)
            width: Target width/tolerance (meters)
            
        Returns:
            Predicted movement time (seconds)
        """
        # Ensure minimum width to avoid log issues
        width = max(width, 1e-6)
        # Ensure minimum distance 
        distance = max(distance, width)
        
        # Fitts' law: MT = a + b * log2(2D/W)
        index_of_difficulty = np.log2(2 * distance / width)
        movement_time = self.a + self.b * index_of_difficulty
        
        # Ensure minimum movement time
        movement_time = max(movement_time, 0.1)
        
        return movement_time
    
    def calculate_index_of_difficulty(self, distance: float, width: float) -> float:
        """
        Calculate index of difficulty (ID).
        
        Args:
            distance: Distance to target (meters)
            width: Target width/tolerance (meters)
            
        Returns:
            Index of difficulty (bits)
        """
        # Ensure minimum width to avoid log issues
        width = max(width, 1e-6)
        # Ensure minimum distance 
        distance = max(distance, width)
        
        return np.log2(2 * distance / width)
    
    def calculate_throughput(self, distance: float, width: float, actual_time: float) -> float:
        """
        Calculate throughput (effective bandwidth).
        
        Args:
            distance: Distance to target (meters)
            width: Target width/tolerance (meters)
            actual_time: Actual movement time (seconds)
            
        Returns:
            Throughput (bits/second)
        """
        id_val = self.calculate_index_of_difficulty(distance, width)
        return id_val / max(actual_time, 1e-6)
    
    def adjust_horizon_for_fitts(self, predicted_time: float, h: float, current_horizon: int) -> int:
        """
        Adjust planning horizon to match Fitts' law prediction.
        
        Args:
            predicted_time: Predicted movement time from Fitts' law (seconds)
            h: Time step (seconds)
            current_horizon: Current horizon length
            
        Returns:
            Adjusted horizon length
        """
        required_steps = int(np.ceil(predicted_time / h)) + 1
        return max(required_steps, current_horizon)  # Don't reduce horizon below current
    
    def get_fitts_parameters_for_task(self, task_meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Extract Fitts' law parameters from task metadata.
        
        Args:
            task_meta: Task metadata dictionary
            
        Returns:
            Dictionary with distance, width, and predicted time
        """
        if task_meta is None:
            return {"distance": None, "width": None, "predicted_time": None}
        
        distance = task_meta.get('distance', None)
        width = task_meta.get('width', None)
        
        if distance is not None and width is not None:
            predicted_time = self.calculate_movement_time(distance, width)
            return {
                "distance": distance,
                "width": width,
                "predicted_time": predicted_time,
                "index_of_difficulty": self.calculate_index_of_difficulty(distance, width)
            }
        
        return {"distance": distance, "width": width, "predicted_time": None}
    
    @classmethod
    def from_empirical_data(cls, movement_times: np.ndarray, distances: np.ndarray, 
                           widths: np.ndarray) -> 'FittsLaw':
        """
        Create Fitts' law model from empirical data using linear regression.
        
        Args:
            movement_times: Observed movement times (seconds)
            distances: Corresponding distances (meters)
            widths: Corresponding target widths (meters)
            
        Returns:
            FittsLaw instance with fitted parameters
        """
        # Calculate index of difficulty for each data point
        ids = np.log2(2 * distances / np.maximum(widths, 1e-6))
        
        # Linear regression: MT = a + b * ID
        # Using normal equations: [a, b]^T = (X^T X)^{-1} X^T y
        X = np.column_stack([np.ones(len(ids)), ids])
        y = movement_times
        
        try:
            params = np.linalg.solve(X.T @ X, X.T @ y)
            a, b = params[0], params[1]
            
            # Ensure reasonable bounds
            a = max(a, 0.01)  # Minimum intercept
            b = max(b, 0.01)  # Minimum slope
            
            return cls(a=a, b=b)
        except np.linalg.LinAlgError:
            # Fallback to default parameters if regression fails
            return cls()
    
    def __repr__(self) -> str:
        return f"FittsLaw(a={self.a:.3f}, b={self.b:.3f})"

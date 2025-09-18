from dataclasses import dataclass
import numpy as np

@dataclass
class KukaLBRIIWA7Model:
    """
    Dynamics model for the KUKA LBR IIWA 7 R800 robot.
    Extracted from the URDF file.
    """
    name: str = "kuka_lbr_iiwa7_r800"
    dof: int = 7
    
    # Link masses (kg) from URDF
    masses: np.ndarray = None
    
    # Link inertias (kg*m^2) - diagonal elements from URDF
    inertias: np.ndarray = None
    
    # Joint limits from URDF (radians)
    joint_limits_lower: np.ndarray = None
    joint_limits_upper: np.ndarray = None
    
    # Joint effort limits (Nm) from URDF
    joint_effort_limits: np.ndarray = None
    
    # Joint velocity limits (rad/s) from URDF
    joint_velocity_limits: np.ndarray = None
    
    # Joint damping coefficients (estimated, not in URDF)
    joint_damping: np.ndarray = None
    
    # DH parameters or joint transforms from URDF
    # Using the joint origins from URDF
    joint_origins: list = None
    joint_axes: list = None
    
    @classmethod
    def default(cls):
        """Create default KUKA LBR IIWA 7 R800 model with parameters from URDF."""
        
        # Masses from URDF inertial properties
        masses = np.array([
            21.163,  # link1
            25.140,  # link2
            21.163,  # link3
            25.140,  # link4
            9.535,   # link5
            14.201,  # link6
            2.472    # link7
        ])
        
        # Diagonal inertia elements (simplified - using average of Ixx, Iyy, Izz)
        inertias = np.array([
            0.102,  # link1
            0.148,  # link2
            0.102,  # link3
            0.148,  # link4
            0.041,  # link5
            0.034,  # link6
            0.002   # link7
        ])
        
        # Joint limits from URDF (radians)
        joint_limits_lower = np.array([
            -2.97,  # joint1
            -2.09,  # joint2
            -2.97,  # joint3
            -2.09,  # joint4
            -2.97,  # joint5
            -2.09,  # joint6
            -3.05   # joint7
        ])
        
        joint_limits_upper = np.array([
            2.97,   # joint1
            2.09,   # joint2
            2.97,   # joint3
            2.09,   # joint4
            2.97,   # joint5
            2.09,   # joint6
            3.05    # joint7
        ])
        
        # Joint effort limits from URDF (Nm)
        joint_effort_limits = np.array([
            176,  # joint1
            176,  # joint2
            110,  # joint3
            110,  # joint4
            110,  # joint5
            40,   # joint6
            40    # joint7
        ])
        
        # Joint velocity limits from URDF (rad/s)
        joint_velocity_limits = np.array([
            1.71,  # joint1
            1.71,  # joint2
            1.75,  # joint3
            2.27,  # joint4
            2.44,  # joint5
            3.14,  # joint6
            3.14   # joint7
        ])
        
        # Estimated joint damping coefficients (not in URDF)
        # Using scaled values based on joint effort limits
        joint_damping = joint_effort_limits * 0.002
        
        # Joint origins and axes from URDF for forward kinematics
        # Format: [(x, y, z, roll, pitch, yaw)]
        joint_origins = [
            (0, 0, 0.3375, 0, 0, 0),           # joint1
            (0, 0, 0, -np.pi/2, 0, 0),         # joint2
            (0, -0.3993, 0, np.pi/2, 0, 0),    # joint3
            (0, 0, 0, -np.pi/2, 0, 0),         # joint4
            (0, -0.3993, 0, np.pi/2, 0, 0),    # joint5
            (0, 0, 0, -np.pi/2, 0, 0),         # joint6
            (0, -0.126, 0, np.pi/2, 0, 0)      # joint7
        ]
        
        # Joint axes from URDF (in local frame)
        joint_axes = [
            (0, 0, -1),  # joint1
            (0, 0, 1),   # joint2
            (0, 0, -1),  # joint3
            (0, 0, -1),  # joint4
            (0, 0, -1),  # joint5
            (0, 0, 1),   # joint6
            (0, 0, 1)    # joint7
        ]
        
        return cls(
            masses=masses,
            inertias=inertias,
            joint_limits_lower=joint_limits_lower,
            joint_limits_upper=joint_limits_upper,
            joint_effort_limits=joint_effort_limits,
            joint_velocity_limits=joint_velocity_limits,
            joint_damping=joint_damping,
            joint_origins=joint_origins,
            joint_axes=joint_axes
        )
"""
Main planner class that orchestrates all modules for human-like motion planning.
"""

from __future__ import annotations

import os
import numpy as np
import casadi as ca
from typing import Optional, Dict, Any, List, Union
from types import SimpleNamespace

from ..dynamics import RobotKinematics, RobotDynamics
from ..solver import SolverOptions, BoundsComputer
from ..costs import CostFunctions, PostSolveCostAnalysis
from ..fitts import FittsLaw
from ..trajectory import InitialTrajectoryGenerator

# Import parameter classes
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from parameters.dynamics.kuka_lbr_iiwa7 import KukaLBRIIWA7Model
from parameters.rewards.van_hall_reward_3d import VanHallRewardParams3D
from parameters.tasks.default_task_3d import TaskParams3D


class VanHallHumanReaching3D_Optimized:

    
    def __init__(self, 
                 H: int = 30,
                 reward_params: Optional[VanHallRewardParams3D] = None,
                 robot_model: Optional[KukaLBRIIWA7Model] = None,
                 cov_min: float = 1e-6,
                 nlp_opts: Optional[Dict] = None,
                 jitter: float = 1e-12,
                 solver_type: str = "mumps",
                 urdf_path: Optional[str] = None):

        # Constants
        self.nq = 7   # 7 DOF robot
        self.NST = 14  # state vector [q1...q7, dq1...dq7]
        self.NGOAL = 3 # 3D goal space
        
        # Parameters
        self.H = int(H)
        self.rp = reward_params or VanHallRewardParams3D.default()
        self.model = robot_model or KukaLBRIIWA7Model.default()
        self.cov_min = float(cov_min)
        self.jitter = float(jitter)
        self.solver_type = solver_type
        
        # Initialize modules
        self._init_modules()
        
        # Configure solver options
        self.nlp_opts = SolverOptions.get_solver_options(solver_type, nlp_opts)
        
        
        # Pre-build solver structure (done once)
        self._build_solver_template()
    
    def _init_modules(self):
        """Initialize all planning modules."""
        # Initialize kinematics and dynamics
        self.kinematics = RobotKinematics()
        # Use joint damping from URDF if available, otherwise fall back to model
        joint_damping = getattr(self.kinematics, 'joint_damping', np.array(self.model.joint_damping))
        self.dynamics = RobotDynamics(self.kinematics, joint_damping)
        
        # Initialize trajectory generator
        self.trajectory_generator = InitialTrajectoryGenerator(self.kinematics, self.dynamics)
        
        # Initialize cost functions
        self.cost_functions = CostFunctions(
            self.kinematics.fwd_kin, 
            self.kinematics.jac_fn,
            self.H, self.nq, self.NST, self.NGOAL
        )
        
        # Initialize bounds computer - use limits from URDF if available, otherwise fall back to model
        joint_limits_lower = getattr(self.kinematics, 'joint_limits_lower', np.array(self.model.joint_limits_lower))
        joint_limits_upper = getattr(self.kinematics, 'joint_limits_upper', np.array(self.model.joint_limits_upper))
        joint_velocity_limits = getattr(self.kinematics, 'joint_velocity_limits', np.array(self.model.joint_velocity_limits))
        joint_effort_limits = getattr(self.kinematics, 'joint_effort_limits', np.array(self.model.joint_effort_limits))
        
        self.bounds_computer = BoundsComputer(
            self.H, self.nq, self.NST, self.NGOAL, self.cov_min,
            joint_limits_lower, joint_limits_upper,
            joint_velocity_limits, joint_effort_limits
        )
        
        # Initialize Fitts' law
        self.fitts_law = FittsLaw(self.rp.fitts_a, self.rp.fitts_b)
        
        # Initialize post-solve analysis
        self.post_solve_analysis = PostSolveCostAnalysis(
            self.kinematics.fwd_kin, self.kinematics.jac_fn, self.H, self.nq
        )
    
    def _build_solver_template(self):
        """Pre-build the solver structure once (major optimization)."""
        H = self.H

        # Decision variables
        X = ca.MX.sym('X', self.NST, H)   # Mean trajectory [mu]
        C = ca.MX.sym('C', self.NST, H)   # Covariance diagonal [cov]  
        U = ca.MX.sym('U', self.nq, H)    # Control trajectory [tau]
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(C, -1, 1), ca.reshape(U, -1, 1))

        # Parameters (will be updated per solve)
        p = ca.MX.sym('p', self.NST + self.NGOAL + self.nq*self.nq + self.nq*self.nq + 1 + 1 + self.NGOAL)
        xi0 = p[0:self.NST]  # Initial state
        goal = p[self.NST:self.NST+self.NGOAL]  # Goal position
        M_inv_vec = p[self.NST+self.NGOAL:self.NST+self.NGOAL+self.nq*self.nq]
        D_vec = p[self.NST+self.NGOAL+self.nq*self.nq:self.NST+self.NGOAL+2*self.nq*self.nq]
        fitts_duration = p[self.NST+self.NGOAL+2*self.nq*self.nq]  # Fitts' law duration
        task_width = p[self.NST+self.NGOAL+2*self.nq*self.nq+1]  # Task width parameter
        start_ee_pos = p[self.NST+self.NGOAL+2*self.nq*self.nq+2:self.NST+self.NGOAL+2*self.nq*self.nq+2+self.NGOAL]  # Actual starting EE position
        M_inv = ca.reshape(M_inv_vec, self.nq, self.nq)
        D = ca.reshape(D_vec, self.nq, self.nq)

        # Get linearized dynamics matrices
        h = self.rp.h
        sigma_tau = self.rp.sigma_tau
        
        # Build dynamics constraints
        g = self._build_dynamics_constraints(X, C, U, M_inv, D, h, sigma_tau)
        
        # Add other constraints
        g.extend(self._build_other_constraints(X, C, U, xi0, goal, fitts_duration, start_ee_pos))
        
        # Stack all constraints
        g = ca.vertcat(*g)
        
        # Build cost function
        cost_types = self.rp.cost_type if isinstance(self.rp.cost_type, list) else [self.rp.cost_type]
        cost_params = self._get_cost_params()
        
        J = self.cost_functions.compute_total_cost(
            X, C, U, xi0, goal, task_width, cost_types, cost_params, h
        )
        
        # Compute bounds
        lbx, ubx = self.bounds_computer.compute_variable_bounds()
        lbg, ubg = self.bounds_computer.compute_constraint_bounds()

        # Store template (built once, reused)
        self.prob_template = {'f': J, 'x': w, 'g': g, 'p': p}
        self.args_template = {'lbx': lbx, 'ubx': ubx, 'lbg': lbg, 'ubg': ubg}
        
        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', self.prob_template, self.nlp_opts)
    
    def _build_dynamics_constraints(self, X, C, U, M_inv, D, h, sigma_tau):
        """Build dynamics constraints."""
        # Implicit damping, explicit position linearization using parameters M_inv, D
        Iq_ca = ca.DM(np.eye(self.nq))
        Oq_ca = ca.DM(np.zeros((self.nq, self.nq)))
        V_update_inv = ca.solve(Iq_ca + h * (M_inv @ D), Iq_ca)  # (I + h M^{-1} D)^{-1}
        A_top = ca.horzcat(Iq_ca, h * Iq_ca)
        A_bottom = ca.horzcat(Oq_ca, V_update_inv)
        A = ca.vertcat(A_top, A_bottom)
        B = ca.vertcat(Oq_ca, h * V_update_inv @ M_inv)

        # Vectorized dynamics constraints
        Xk = X[:, 0:self.H-1]  # Current states
        Uk = U[:, 0:self.H-1]  # Current controls
        Ck = C[:, 0:self.H-1]  # Current covariances
        X_next = X[:, 1:self.H]  # Next states
        C_next = C[:, 1:self.H]  # Next covariances

        # Mean dynamics: mu_next = A @ mu + B @ u (vectorized)
        mu_next_pred = ca.mtimes(A, Xk) + ca.mtimes(B, Uk)
        g_mean = mu_next_pred - X_next

        # Covariance dynamics (vectorized)
        # For diagonal covariance: diag(A @ diag(cov) @ A.T) = (A^2) @ cov (element-wise)
        A_squared = ca.power(A, 2)  # Element-wise square
        cov_from_A = A_squared @ Ck
        
        # Control contribution: torque derivative dependent noise
        if self.H > 2:  # Need at least 2 control inputs for derivatives
            U_diff = Uk[:, 1:] - Uk[:, :-1]  # Torque derivatives
            # Apply to dynamics from k=1 to k=H-2
            cov_from_B_deriv = (1 + sigma_tau) * ca.power(B @ U_diff, 2)
            # Pad with zeros for first timestep
            cov_from_B = ca.horzcat(ca.DM.zeros(self.NST, 1), cov_from_B_deriv)
        else:
            cov_from_B = ca.DM.zeros(self.NST, self.H-1)
        
        cov_next_pred = cov_from_A + cov_from_B 
        g_cov = cov_next_pred - C_next

        g_dynamics_flat = ca.vertcat(ca.reshape(g_mean, -1, 1), ca.reshape(g_cov, -1, 1))
        
        return [g_dynamics_flat]
    
    def _build_other_constraints(self, X, C, U, xi0, goal, fitts_duration, start_ee_pos=None):
        """Build other constraints (initial, final, etc.)."""
        g = []
        
        # Initial condition constraint
        g.append(xi0 - X[:, 0])
        
        # Initial covariance constraint
        g.append(1e-5 * ca.DM.ones(self.NST, 1) - C[:, 0])
        
        # Initial control constraint - zero input at start
        g.append(U[:, 0])

        # Final constraint - end-effector reaches goal
        q_final = X[:self.nq, -1]
        ee_final = self.kinematics.forward_kinematics_symbolic(q_final)
        g.append(ee_final - goal)
        
        # Final velocity constraint - velocity zero at goal
        dq_final = X[self.nq:self.NST, -1]
        g.append(dq_final)

        # Fitts' law final time constraint (if duration provided)
        final_time = (self.H - 1) * self.rp.h  # Total trajectory time
        fitts_constraint = ca.if_else(fitts_duration > 0, final_time - fitts_duration, 0)
        g.append(fitts_constraint)

        # Covariance positivity constraints
        g.append(ca.reshape(C, -1, 1))
        
        return g
    
    def _get_cost_params(self) -> Dict[str, float]:
        """Get cost parameters dictionary."""
        return {
            "ctrl_cost": self.rp.ctrl_cost,
            "input_adapted_cost": self.rp.input_adapted_cost,
            "torque_derivative_cost": self.rp.torque_derivative_cost,
            "distance_cost": self.rp.distance_cost,
            "path_straightness_cost": self.rp.path_straightness_cost,
            "end_effector_covariance_cost": self.rp.end_effector_covariance_cost,
            "endpoint_jerk_cost": self.rp.endpoint_jerk_cost,
            "use_terminal_cost": self.rp.use_terminal_cost,
            "rescale": self.rp.rescale,
        }
    
    def solve(self, task: TaskParams3D, task_meta: Dict = None, start_ee_pos: np.ndarray = None) -> SimpleNamespace:
        """
        Solve the NLP using pre-built solver template (major optimization).
        
        Args:
            task: Task parameters
            task_meta: Task metadata (distance, width, etc.)
            start_ee_pos: Actual starting EE position (if None, computed from initial joint angles)
            
        Returns:
            Optimization result with cost analysis
        """
        # Convert initial state and goal
        xi0 = np.asarray(task.xi0, dtype=float).reshape(self.NST)
        goal = np.asarray(task.goal, dtype=float).reshape(self.NGOAL)

        # Calculate Fitts' law duration if enabled
        fitts_duration = None
        width = task_meta.get('width', task.width) if task_meta is not None else task.width
        
        if self.rp.use_fitts_law:
            fitts_params = self.fitts_law.get_fitts_parameters_for_task(task_meta)
            if fitts_params["predicted_time"] is not None:
                fitts_duration = fitts_params["predicted_time"]
                print(f"Fitts' law: D={fitts_params['distance']:.3f}m, W={width:.3f}m, MT={fitts_duration:.3f}s")
                
                # Adjust horizon to match Fitts' duration
                new_H = self.fitts_law.adjust_horizon_for_fitts(fitts_duration, self.rp.h, self.H)
                if new_H != self.H:
                    print(f"Adjusting horizon from {self.H} to {new_H} steps for Fitts' duration")
                    old_H = self.H
                    self.H = new_H
                    
                    # Update bounds computer with new horizon
                    self.bounds_computer = BoundsComputer(
                        self.H, self.nq, self.NST, self.NGOAL, self.cov_min,
                        np.array(self.model.joint_limits_lower),
                        np.array(self.model.joint_limits_upper),
                        np.array(self.model.joint_velocity_limits),
                        np.array(self.model.joint_effort_limits)
                    )
                    
                    # Rebuild solver template with new horizon
                    print("Rebuilding solver template for new horizon...")
                    self._build_solver_template()
                    print(f"Solver rebuilt: {old_H} → {self.H} steps")

        # Compute mass matrix and damping at initial configuration
        q0 = xi0[:self.nq]
        M = self.dynamics.compute_mass_matrix(q0)
        M_inv = np.linalg.inv(M + 0.2 * np.eye(self.nq))  # Add regularization
        D = np.diag(self.dynamics.joint_damping)
        

        start_ee_pos = np.asarray(start_ee_pos, dtype=float).reshape(self.NGOAL)

        # Combine parameters
        fitts_value = fitts_duration if fitts_duration is not None else 0.0
        p = np.concatenate([
            xi0, 
            goal, 
            M_inv.ravel(order='F'), 
            D.ravel(order='F'),
            np.array([fitts_value]),
            np.array([width]),
            start_ee_pos
        ])
        
        # Create initial guess
        x0 = self._create_initial_guess(task, fitts_duration)
        
        # Solve using pre-built solver
        sol = self.solver(p=p, x0=x0, **self.args_template)
        
        # Extract solution
        w_opt = np.array(sol['x']).squeeze()
        
        # Parse solution
        result = self._parse_solution(w_opt)
        
        # Add cost analysis
        cost_types = self.rp.cost_type if isinstance(self.rp.cost_type, list) else [self.rp.cost_type]
        cost_params = self._get_cost_params()
        
        result.cost_terms = self.post_solve_analysis.analyze_costs(
            result, task, cost_types, cost_params, self.rp.h, width
        )
        
        return result
    
    def _create_initial_guess(self, task: TaskParams3D, fitts_duration: Optional[float] = None) -> np.ndarray:
        
        # Create simple initial guess
        x0 = self.trajectory_generator.create_simple_initial_guess(task, self.H, use_ik=True)
        print("Created new initial guess with IK")
        return x0
    
    def _parse_solution(self, w_opt: np.ndarray) -> SimpleNamespace:
        """Parse optimization solution into result structure."""
        H = self.H 
        nX = self.NST * H
        nC = self.NST * H
        nU = self.nq * H

        Xo = w_opt[0:nX].reshape(self.NST, H, order='F')
        Co = w_opt[nX:nX+nC].reshape(self.NST, H, order='F')
        Uo = w_opt[nX+nC:nX+nC+nU].reshape(self.nq, H, order='F')

        # Create result
        res = SimpleNamespace()
        res.mu = Xo
        res.cov = Co
        res.tau = Uo
        
        # For compatibility
        res.q = res.mu[:self.nq, :]
        res.dq = res.mu[self.nq:self.NST, :]
        res.t = np.arange(H) * self.rp.h
        
        # Add end-effector positions using same FK as optimization constraints
        res.ee_positions = np.zeros((self.NGOAL, H))
        for i in range(H):
            # Use the same FK method as the optimization constraints for consistency
            ee_pos_internal = np.array(self.kinematics.fwd_kin(res.q[:, i])).ravel()
            # Transform to controller frame for display/interface consistency
            ee_pos_controller = self.kinematics.transform_to_controller_frame(ee_pos_internal)
            res.ee_positions[:, i] = ee_pos_controller

        return res
    

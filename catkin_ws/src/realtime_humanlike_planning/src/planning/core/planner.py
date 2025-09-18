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
    """
    Optimized version of VanHallHumanReaching3D with significant performance improvements:
    1. Reduced horizon length (30 instead of 150)
    2. Pre-built solver structure (built once, not per solve)
    3. CasADi maps for vectorized operations
    4. Frozen linearization for efficiency
    5. Maintains full nonlinear dynamics for accuracy
    6. Uses IPOPT with MUMPS linear solver for smooth human-like trajectories
    """
    
    def __init__(self, 
                 H: int = 30,
                 reward_params: Optional[VanHallRewardParams3D] = None,
                 robot_model: Optional[KukaLBRIIWA7Model] = None,
                 cov_min: float = 1e-6,
                 nlp_opts: Optional[Dict] = None,
                 jitter: float = 1e-12,
                 use_prev_traj_warm_start: bool = False,
                 solver_type: str = "mumps",
                 urdf_path: Optional[str] = None):
        """
        Initialize the optimized planner.
        
        Args:
            H: Horizon length (reduced from 150 to 30 for speed)
            reward_params: Reward parameters
            robot_model: Robot model parameters
            cov_min: Minimum covariance value
            nlp_opts: Custom NLP solver options
            jitter: Numerical jitter for stability
            use_prev_traj_warm_start: Whether to use warm starting
            solver_type: Solver type ("mumps", "ma27", "ma57")
            urdf_path: Path to URDF file
        """
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
        self.use_prev_traj_warm_start = bool(use_prev_traj_warm_start)
        self.solver_type = solver_type
        
        # Warm starting capability
        self.previous_solution = None
        self.solution_cache = {}  # Cache solutions for similar tasks
        
        # Initialize modules
        self._init_modules(urdf_path)
        
        # Configure solver options
        self.nlp_opts = SolverOptions.get_solver_options(solver_type, nlp_opts)
        
        # Pre-build solver structure (done once)
        self._build_solver_template()
    
    def _init_modules(self, urdf_path: Optional[str]):
        """Initialize all planning modules."""
        # Initialize kinematics and dynamics
        self.kinematics = RobotKinematics(urdf_path)
        self.dynamics = RobotDynamics(self.kinematics, np.array(self.model.joint_damping))
        
        # Initialize trajectory generator
        self.trajectory_generator = InitialTrajectoryGenerator(self.kinematics, self.dynamics)
        
        # Initialize cost functions
        self.cost_functions = CostFunctions(
            self.kinematics.fwd_kin, 
            self.kinematics.jac_fn,
            self.H, self.nq, self.NST, self.NGOAL
        )
        
        # Initialize bounds computer
        self.bounds_computer = BoundsComputer(
            self.H, self.nq, self.NST, self.NGOAL, self.cov_min,
            np.array(self.model.joint_limits_lower),
            np.array(self.model.joint_limits_upper),
            np.array(self.model.joint_velocity_limits),
            np.array(self.model.joint_effort_limits)
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
        p = ca.MX.sym('p', self.NST + self.NGOAL + self.nq*self.nq + self.nq*self.nq + 1 + 1)
        xi0 = p[0:self.NST]  # Initial state
        goal = p[self.NST:self.NST+self.NGOAL]  # Goal position
        M_inv_vec = p[self.NST+self.NGOAL:self.NST+self.NGOAL+self.nq*self.nq]
        D_vec = p[self.NST+self.NGOAL+self.nq*self.nq:self.NST+self.NGOAL+2*self.nq*self.nq]
        fitts_duration = p[self.NST+self.NGOAL+2*self.nq*self.nq]  # Fitts' law duration
        task_width = p[self.NST+self.NGOAL+2*self.nq*self.nq+1]  # Task width parameter
        M_inv = ca.reshape(M_inv_vec, self.nq, self.nq)
        D = ca.reshape(D_vec, self.nq, self.nq)

        # Get linearized dynamics matrices
        h = self.rp.h
        sigma_tau = self.rp.sigma_tau
        
        # Build dynamics constraints
        g = self._build_dynamics_constraints(X, C, U, M_inv, D, h, sigma_tau)
        
        # Add other constraints
        g.extend(self._build_other_constraints(X, C, U, xi0, goal, fitts_duration))
        
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
    
    def _build_other_constraints(self, X, C, U, xi0, goal, fitts_duration):
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
    
    def solve(self, task: TaskParams3D, task_meta: Dict = None) -> SimpleNamespace:
        """
        Solve the NLP using pre-built solver template (major optimization).
        
        Args:
            task: Task parameters
            task_meta: Task metadata (distance, width, etc.)
            
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
                    # Rebuild solver template with new horizon
                    print("Rebuilding solver template for new horizon...")
                    self._build_solver_template()
                    print(f"Solver rebuilt: {old_H} → {self.H} steps")

        # Compute mass matrix and damping at initial configuration
        q0 = xi0[:self.nq]
        M = self.dynamics.compute_mass_matrix(q0)
        M_inv = np.linalg.inv(M + 0.2 * np.eye(self.nq))  # Add regularization
        D = np.diag(self.dynamics.joint_damping)

        # Combine parameters
        fitts_value = fitts_duration if fitts_duration is not None else 0.0
        p = np.concatenate([
            xi0, 
            goal, 
            M_inv.ravel(order='F'), 
            D.ravel(order='F'),
            np.array([fitts_value]),
            np.array([width])
        ])
        
        # Create initial guess
        x0 = self._create_initial_guess(task, fitts_duration)
        
        # Solve using pre-built solver
        sol = self.solver(p=p, x0=x0, **self.args_template)
        
        # Extract solution and cache for warm starting
        w_opt = np.array(sol['x']).squeeze()
        self.previous_solution = w_opt.copy()
        
        # Cache solution for similar tasks
        task_key = self._get_task_key(task)
        self.solution_cache[task_key] = w_opt.copy()
        
        # Limit cache size
        if len(self.solution_cache) > 10:
            oldest_key = next(iter(self.solution_cache))
            del self.solution_cache[oldest_key]
        
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
        """Create initial guess for optimization."""
        # Try warm starting if enabled
        if self.use_prev_traj_warm_start and self.previous_solution is not None:
            adapted = self.trajectory_generator.create_warm_start_from_previous(
                self.previous_solution, task, self.H
            )
            if adapted is not None:
                print("Using warm start from previous solution")
                return adapted
        
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
        
        # Add end-effector positions
        res.ee_positions = np.zeros((self.NGOAL, H))
        for i in range(H):
            res.ee_positions[:, i] = np.array(self.kinematics.fwd_kin(res.q[:, i])).ravel()

        return res
    
    def _get_task_key(self, task: TaskParams3D) -> str:
        """Generate a key for task similarity matching."""
        q0 = task.xi0[:7]
        goal = task.goal
        return str(hash((tuple(np.round(q0, 2)), tuple(np.round(goal, 2)))))
    
    def plot_temporal_scaling_analysis(self, result: SimpleNamespace, task_params: TaskParams3D, 
                                     save_path: Optional[str] = None, show_plot: bool = False):
        """Plot temporal scaling analysis."""
        import matplotlib.pyplot as plt
        
        if not hasattr(result, 'cost_terms') or 'temporal_scaling' not in result.cost_terms:
            raise ValueError("Result must contain temporal_scaling data. Run solve() first.")
        
        scaling_data = result.cost_terms['temporal_scaling']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        time_steps = scaling_data['time_steps']
        
        # Plot 1: Temporal scaling values over time
        ax1.plot(time_steps, scaling_data['scaling_values'], 'b-', linewidth=2, label='Scaling Factor')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Scaling Factor s(t)')
        ax1.set_title(f'Temporal Scaling Evolution (Width={scaling_data["task_width"]:.3f})')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add text annotation
        d_scale = scaling_data.get('d_scale', 'N/A')
        A_scale = scaling_data.get('A_scale', 'N/A')
        equation_text = (f'$s(t) = 1 - A \\cdot \\frac{{t^d \\cdot (1-t)^{{1/d}}}}{{B(d+1, 1/d+1)}}$\n'
                        f'A={A_scale:.3f}, d={d_scale:.3f}')
        ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                fontsize=9)
        
        # Plot 2: Control input changes over time
        ax2.plot(time_steps, scaling_data['control_changes'], 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Input Change Magnitude')
        ax2.set_title('Control Input Changes Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scaled costs over time
        ax3.plot(time_steps, scaling_data['scaled_costs'], 'g-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Scaled Cost Contribution')
        ax3.set_title('Temporal Scaled Cost Contributions')
        ax3.grid(True, alpha=0.3)
        
        total_cost = scaling_data['total_scaled_cost']
        ax3.text(0.05, 0.95, f'Total Scaled Cost: {total_cost:.6f}', 
                transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # Plot 4: Scaling factor vs normalized time step
        step_norm = scaling_data['step_normalized']
        ax4.plot(step_norm, scaling_data['scaling_values'], 'purple', linewidth=2, marker='o', markersize=4)
        ax4.set_xlabel('Normalized Time Step')
        ax4.set_ylabel('Scaling Factor s(t)')
        ax4.set_title('Scaling Factor vs Normalized Time')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal scaling analysis plot saved to {save_path}")
        
        if show_plot:
            plt.show()
            
        return fig

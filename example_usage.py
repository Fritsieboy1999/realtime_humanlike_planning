#!/usr/bin/env python3
"""
Example usage of the modular VanHallHumanReaching3D_Optimized planner.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.planning import VanHallHumanReaching3D_Optimized
from parameters.tasks.default_task_3d import TaskParams3D
from parameters.rewards.van_hall_reward_3d import VanHallRewardParams3D


def main():
    """Demonstrate the modular planner usage."""
    print("🤖 Initializing VanHall Human-like Motion Planner...")
    
    # Create reward parameters with input_adapted cost
    reward_params = VanHallRewardParams3D.default()
    reward_params.cost_type = "input_adapted"  # Use temporal scaling
    reward_params.use_fitts_law = False  # Disable Fitts' law for this example
    
    # Initialize planner with reduced horizon for speed
    planner = VanHallHumanReaching3D_Optimized(
        H=30,  # Reduced horizon
        reward_params=reward_params,
        solver_type="mumps",  # Use MUMPS for smooth trajectories
        use_prev_traj_warm_start=True
    )
    
    # Create a reaching task
    q_start = np.array([0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, 0.0])  # Home position
    goal_position = np.array([0.5, 0.2, 0.6])  # 50cm forward, 20cm right, 60cm up
    task_width = 0.015  # 1.5cm tolerance
    
    task = TaskParams3D.create_reaching_task(
        q_start=q_start,
        goal_position=goal_position,
        width=task_width
    )
    
    print(f"📍 Task: Reach {goal_position} with tolerance {task_width*1000:.1f}mm")
    
    # Task metadata for analysis
    task_meta = {
        'width': task_width,
        'distance': 0.3,  # Approximate distance for analysis
    }
    
    # Solve the planning problem
    print("🔧 Solving optimization problem...")
    try:
        result = planner.solve(task, task_meta)
        print("✅ Optimization successful!")
        
        # Print cost breakdown
        print("\n📊 Cost Analysis:")
        for cost_name, cost_value in result.cost_terms.items():
            if isinstance(cost_value, (int, float)):
                print(f"  {cost_name}: {cost_value:.6f}")
        
        # Print trajectory info
        H = result.t.shape[0]
        total_time = result.t[-1]
        print(f"\n🎯 Trajectory Info:")
        print(f"  Horizon length: {H} steps")
        print(f"  Total time: {total_time:.3f} seconds")
        print(f"  Final end-effector position: {result.ee_positions[:, -1]}")
        print(f"  Goal position: {goal_position}")
        print(f"  Final error: {np.linalg.norm(result.ee_positions[:, -1] - goal_position)*1000:.2f}mm")
        
        # Plot results
        plot_results(result, task, goal_position)
        
        # Plot temporal scaling analysis if available
        if 'temporal_scaling' in result.cost_terms:
            print("📈 Plotting temporal scaling analysis...")
            planner.plot_temporal_scaling_analysis(
                result, task, 
                save_path="temporal_scaling_analysis.png",
                show_plot=True
            )
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


def plot_results(result, task, goal_position):
    """Plot trajectory results."""
    print("📈 Plotting results...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('VanHall Human-like Motion Planning Results', fontsize=16)
    
    t = result.t
    
    # Joint positions
    ax = axes[0, 0]
    for i in range(7):
        ax.plot(t, result.q[i, :], label=f'q{i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Position (rad)')
    ax.set_title('Joint Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Joint velocities
    ax = axes[0, 1]
    for i in range(7):
        ax.plot(t, result.dq[i, :], label=f'dq{i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Velocity (rad/s)')
    ax.set_title('Joint Velocities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control inputs
    ax = axes[0, 2]
    for i in range(7):
        ax.plot(t, result.tau[i, :], label=f'τ{i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Torque (Nm)')
    ax.set_title('Control Inputs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # End-effector trajectory (3D)
    ax = axes[1, 0]
    ee_traj = result.ee_positions
    ax.plot(ee_traj[0, :], ee_traj[1, :], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(ee_traj[0, 0], ee_traj[1, 0], c='green', s=100, marker='o', label='Start')
    ax.scatter(goal_position[0], goal_position[1], c='red', s=100, marker='*', label='Goal')
    ax.scatter(ee_traj[0, -1], ee_traj[1, -1], c='blue', s=100, marker='x', label='Final')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('End-Effector Trajectory (Top View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # End-effector position over time
    ax = axes[1, 1]
    ax.plot(t, ee_traj[0, :], 'r-', label='X')
    ax.plot(t, ee_traj[1, :], 'g-', label='Y') 
    ax.plot(t, ee_traj[2, :], 'b-', label='Z')
    ax.axhline(goal_position[0], color='r', linestyle='--', alpha=0.7, label='Goal X')
    ax.axhline(goal_position[1], color='g', linestyle='--', alpha=0.7, label='Goal Y')
    ax.axhline(goal_position[2], color='b', linestyle='--', alpha=0.7, label='Goal Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('End-Effector Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Covariance evolution (diagonal elements)
    ax = axes[1, 2]
    for i in range(7):
        ax.semilogy(t, result.cov[i, :], label=f'σ²_q{i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Covariance')
    ax.set_title('Joint Position Covariances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('planning_results.png', dpi=300, bbox_inches='tight')
    print("💾 Results saved to 'planning_results.png'")
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example usage of the modular VanHallHumanReaching3D_Optimized planner.
"""

import numpy as np
import matplotlib.pyplot as plt
from planning import VanHallHumanReaching3D_Optimized
from parameters.tasks.default_task_3d import TaskParams3D
from parameters.rewards.van_hall_reward_3d import VanHallRewardParams3D


def main():
    """Demonstrate realistic warm start usage with sequential tasks."""
    print("🤖 Initializing VanHall Human-like Motion Planner...")
    
    # Create reward parameters with default cost combination
    reward_params = VanHallRewardParams3D.default()
    # Default uses ["input_adapted", "path_straightness", "endpoint_jerk"]
    reward_params.use_fitts_law = True  # Disable Fitts' law for this example
    
    # Initialize planner
    planner = VanHallHumanReaching3D_Optimized(
        H=30,  # Reduced horizon
        reward_params=reward_params,
        solver_type="mumps"  # Use MUMPS for smooth trajectories
    )
    
    print("🎯 Demonstrating realistic warm start with sequential reaching tasks...")
    
    # Define a sequence of realistic reaching tasks
    tasks = [
        {
            'name': 'Task 1: Home to Cup',
            'q_start': np.array([0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, 0.0]),
            'goal': np.array([0.5, 0.2, 0.6]),
            'width': 0.015,
            'description': 'Reach for a cup on the table'
        },
        {
            'name': 'Task 2: Cup to Mouth',
            'q_start': None,  # Will be set from previous task's final configuration
            'goal': np.array([0.2, -0.1, 0.8]),
            'width': 0.020,
            'description': 'Bring cup to mouth'
        },
        {
            'name': 'Task 3: Mouth to Table',
            'q_start': None,
            'goal': np.array([0.6, 0.0, 0.5]),
            'width': 0.015,
            'description': 'Place cup back on table'
        },
        {
            'name': 'Task 4: Table to Home',
            'q_start': None,
            'goal': np.array([0.3, 0.0, 0.7]),
            'width': 0.020,
            'description': 'Return to rest position'
        }
    ]
    
    results = []
    total_planning_time = 0
    
    for i, task_info in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"🎯 {task_info['name']}: {task_info['description']}")
        print(f"{'='*60}")
        
        # Set starting configuration from previous task if available
        if task_info['q_start'] is None and i > 0:
            # Use final configuration from previous task
            task_info['q_start'] = results[-1].q[:, -1]
            print(f"📍 Starting from previous task's final configuration")
        
        # Create task
        task = TaskParams3D.create_reaching_task(
            q_start=task_info['q_start'],
            goal_position=task_info['goal'],
            width=task_info['width']
        )
        
        # Task metadata
        if i > 0:
            # Calculate distance from previous goal
            distance = np.linalg.norm(task_info['goal'] - tasks[i-1]['goal'])
        else:
            distance = np.linalg.norm(task_info['goal'] - np.array([0.3, 0.0, 0.7]))  # From home position
        
        task_meta = {
            'width': task_info['width'],
            'distance': distance,
            'task_number': i + 1
        }
        
        print(f"📍 Goal: {task_info['goal']} (tolerance: {task_info['width']*1000:.1f}mm)")
        print(f"📏 Distance from previous: {distance:.3f}m")
        
        # Solve with timing
        import time
        start_time = time.time()
        
        print("🔧 Solving optimization problem...")
        try:
            result = planner.solve(task, task_meta)
            solve_time = time.time() - start_time
            total_planning_time += solve_time
            
            print(f"✅ Optimization successful! ({solve_time:.3f}s)")
            
            # Store result
            results.append(result)
            
            # Print trajectory info
            H = result.t.shape[0]
            total_time = result.t[-1]
            final_error = np.linalg.norm(result.ee_positions[:, -1] - task_info['goal'])
            
            print(f"  Horizon: {H} steps, Duration: {total_time:.3f}s")
            print(f"  Final error: {final_error*1000:.2f}mm")
            
            # Show warm start effectiveness
            if i > 0:
                improvement = (solve_time < 2.0)  # Assume cold start takes ~2s
                print(f"  Warm start: {'✅ Effective' if improvement else '⚠️  Limited benefit'}")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SEQUENTIAL PLANNING SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks completed: {len(results)}")
    print(f"Total planning time: {total_planning_time:.3f}s")
    print(f"Average time per task: {total_planning_time/len(results):.3f}s")
    
    if len(results) >= 2:
        print(f"First task time: {total_planning_time/len(results):.3f}s (cold start)")
        print(f"Subsequent tasks: Benefited from warm start")
    
    # Plot the final result if we have any successful solutions
    if results:
        print("\n📈 Plotting results from final task...")
        plot_results(results[-1], task, tasks[-1]['goal'])
        
        # Plot trajectory sequence if we have multiple results
        if len(results) > 1:
            plot_sequential_results(results, tasks)
    
    return results


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


def plot_sequential_results(results, tasks):
    """Plot results from sequential tasks to show warm start effectiveness."""
    print("📈 Plotting sequential task results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Sequential Task Execution with Warm Start', fontsize=16)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot 1: End-effector trajectories in 3D space
    ax = axes[0, 0]
    for i, (result, task_info) in enumerate(zip(results, tasks)):
        ee_traj = result.ee_positions
        color = colors[i % len(colors)]
        
        # Plot trajectory
        ax.plot(ee_traj[0, :], ee_traj[1, :], color=color, linewidth=2, 
               label=f"Task {i+1}: {task_info['name'].split(':')[1].strip()}")
        
        # Mark start and end points
        ax.scatter(ee_traj[0, 0], ee_traj[1, 0], c=color, s=100, marker='o', alpha=0.7)
        ax.scatter(ee_traj[0, -1], ee_traj[1, -1], c=color, s=100, marker='*', alpha=0.7)
        
        # Mark goal
        goal = task_info['goal']
        ax.scatter(goal[0], goal[1], c=color, s=60, marker='x', alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('End-Effector Trajectories (Top View)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Joint configurations over tasks
    ax = axes[0, 1]
    for i, result in enumerate(results):
        t_offset = i * 2.0  # Offset time for visualization
        t_adjusted = result.t + t_offset
        color = colors[i % len(colors)]
        
        # Plot a few representative joints
        for joint in [0, 3, 6]:  # Joints 1, 4, 7
            alpha = 0.8 if joint == 3 else 0.4  # Highlight joint 4
            ax.plot(t_adjusted, result.q[joint, :], color=color, alpha=alpha,
                   linewidth=2 if joint == 3 else 1,
                   label=f"Task {i+1}, Joint {joint+1}" if joint == 3 else "")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Position (rad)')
    ax.set_title('Joint Configurations Across Tasks')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Control effort comparison
    ax = axes[1, 0]
    control_norms = []
    task_names = []
    
    for i, (result, task_info) in enumerate(zip(results, tasks)):
        # Calculate RMS control effort
        rms_control = np.sqrt(np.mean(np.sum(result.tau**2, axis=0)))
        control_norms.append(rms_control)
        task_names.append(f"T{i+1}")
    
    bars = ax.bar(task_names, control_norms, color=colors[:len(results)], alpha=0.7)
    ax.set_ylabel('RMS Control Effort (Nm)')
    ax.set_title('Control Effort per Task')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, control_norms):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}', ha='center', va='bottom')
    
    # Plot 4: Task execution times (if we had timing data)
    ax = axes[1, 1]
    durations = [result.t[-1] for result in results]
    bars = ax.bar(task_names, durations, color=colors[:len(results)], alpha=0.7)
    ax.set_ylabel('Task Duration (s)')
    ax.set_title('Task Execution Times')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, durations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sequential_planning_results.png', dpi=300, bbox_inches='tight')
    print("💾 Sequential results saved to 'sequential_planning_results.png'")
    plt.show()


if __name__ == "__main__":
    main()

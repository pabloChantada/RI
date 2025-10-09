import os
import numpy as np
import matplotlib.pyplot as plt
from env import CustomEnv
from stable_baselines3 import SAC


# ============================================================================
# DIRECTORIES
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(LOGS_DIR, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pos(info):
    """
    Extract agent and target positions from info dict.
    
    Args:
        info: Environment info dictionary
        
    Returns:
        tuple: (agent_position, target_position)
    """
    return info["agent_position"], info["target_position"]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def run_evaluation(env, model, max_steps=30):
    """
    Execute a complete model evaluation.
    
    Runs one episode and collects:
        - Agent trajectory
        - Target trajectory
        - Episode reward
        - Performance metrics
    
    Args:
        env: Gymnasium environment
        model: Trained model
        max_steps: Maximum number of steps (must match env.max_steps)
    
    Returns:
        None (saves trajectory plot to logs directory)
    """
    # Reset environment
    obs, info = env.reset()
    done = False
    traj_agent, traj_target = [], []
    episode_reward = 0
    distances = []

    # Store initial positions
    agent0, target0 = pos(info)
    traj_agent.append(agent0)
    traj_target.append(target0)
    distances.append(info['distance'])

    # Run episode
    step = 0
    while not done and step < max_steps:
        # Get action from model
        action, _ = model.predict(obs)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track episode data
        episode_reward += reward
        agent, target = pos(info)
        traj_agent.append(agent)
        traj_target.append(target)
        distances.append(info['distance'])
        
        # Check termination
        done = bool(terminated or truncated)
        step += 1
    
    # Calculate episode metrics
    episode_info = {
        'final_distance': distances[-1],
        'initial_distance': distances[0],
        'avg_distance': np.mean(distances),
        'min_distance': np.min(distances),
        'success': distances[-1] < 150  # Goal reached
    }
    
    # Plot and save trajectories
    plot_trajectories(traj_agent, traj_target, episode_info, 
                     out_name="evaluation_trajectory.png")


def plot_trajectories(traj_agent, traj_target, episode_info, half=1000, 
                      out_name="trajectories_2d.png"):
    """
    Plot 2D trajectories of agent and target with enhanced visualization.
    
    Creates a single plot showing:
        - Agent trajectory (blue line with markers)
        - Target trajectory (red dashed line)
        - Start positions (green X for agent, red circle for target)
        - End position (blue star for agent)
    
    Args:
        traj_agent: List of agent positions [(x, z), ...]
        traj_target: List of target positions [(x, z), ...]
        episode_info: Dictionary with episode information
        half: Half of environment size for plot limits
        out_name: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    lim = float(half)
    ax.set_xlim(-lim, +lim)
    ax.set_ylim(-lim, +lim)
    ax.set_aspect("equal", adjustable="box")

    # Extract coordinates
    agent_x = [p[0] for p in traj_agent]
    agent_z = [p[1] for p in traj_agent]
    target_x = [p[0] for p in traj_target]
    target_z = [p[1] for p in traj_target]

    # Plot agent trajectory
    ax.plot(agent_x, agent_z, 'b-', alpha=0.6, linewidth=2, 
           label='Agent trajectory')
    
    # Plot target trajectory
    ax.plot(target_x, target_z, 'r--', alpha=0.5, linewidth=2, 
           label='Cylinder trajectory')
    
    # Plot start positions
    ax.scatter(agent_x[0], agent_z[0], marker='x', s=200, 
              color='green', edgecolors='black', linewidths=2,
              label='Agent start', zorder=5)
    ax.scatter(target_x[0], target_z[0], marker='o', s=200,
              color='red', linewidths=3,
              label='Cylinder start', zorder=5)
    
    # Plot end positions
    ax.scatter(agent_x[-1], agent_z[-1], marker='x', s=300,
              color='blue', linewidths=2,
              label='Agent end', zorder=5)
    ax.scatter(target_x[-1], target_z[-1], marker='o', s=300,
            color='red', linewidths=3,
            label='Cylinder end', zorder=5)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add title with episode information
    title = f"2D Trajectories - Evaluation\n"
    title += f"Final distance: {episode_info['final_distance']:.1f} | "
    title += f"Success: {'Yes' if episode_info['success'] else 'No'}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Labels
    plt.xlabel("X", fontweight='bold')
    plt.ylabel("Z", fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    out_path = os.path.join(LOGS_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved at: {out_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create environment with consistent max_steps
    env = CustomEnv(size=1000, max_steps=30)
    
    # Load trained model
    model_path = os.path.join(MODELS_DIR, "sac_cylinder_final.zip")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Make sure to train the model first with train.py")
        exit(1)
    
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, env=env, device="cpu")
    
    # Run evaluation
    run_evaluation(env, model)
    
    # Close environment
    env.close()
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import CustomEnv


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

MAX_STEPS_PER_EPISODE = 30

# Phase 1: Static Target
EPISODES_PHASE1 = 234
TIMESTEPS_PHASE1 = MAX_STEPS_PER_EPISODE * EPISODES_PHASE1  # 7,020 timesteps

# Phase 2: Moving Target
EPISODES_PHASE2 = 100
TIMESTEPS_PHASE2 = MAX_STEPS_PER_EPISODE * EPISODES_PHASE2  # 3,000 timesteps

# Total training
TOTAL_EPISODES = EPISODES_PHASE1 + EPISODES_PHASE2  # 334 episodes
TOTAL_TIMESTEPS = TIMESTEPS_PHASE1 + TIMESTEPS_PHASE2  # 10,020 timesteps

print(f"\n{'='*70}")
print(f"TRAINING CONFIGURATION")
print(f"{'='*70}")
print(f"Steps per episode: {MAX_STEPS_PER_EPISODE}")
print(f"\nPHASE 1 (Static Target - DOES NOT MOVE):")
print(f"  - Episodes: {EPISODES_PHASE1}")
print(f"  - Timesteps: {TIMESTEPS_PHASE1}")
print(f"\nPHASE 2 (Moving Target - MOVES EVERY STEP):")
print(f"  - Episodes: {EPISODES_PHASE2}")
print(f"  - Timesteps: {TIMESTEPS_PHASE2}")
print(f"\nTOTAL:")
print(f"  - Episodes: {TOTAL_EPISODES}")
print(f"  - Timesteps: {TOTAL_TIMESTEPS}")
print(f"{'='*70}\n")


# ============================================================================
# DIRECTORIES
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================================
# EPISODE METRICS CALLBACK
# ============================================================================

class EpisodeMetricsCallback(BaseCallback):
    """
    Custom callback to log metrics per episode.
    
    Tracks:
        - Total reward per episode
        - Distance metrics (initial, final, average)
        - Episode completion
    """
    
    def __init__(self, verbose=0):
        """
        Initialize metrics callback.
        
        Args:
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_distances = []
        self.current_episode_reward = 0
        self.current_episode_distances = []
        
    def _on_step(self):
        """Called at each step of the environment."""
        # Accumulate reward
        self.current_episode_reward += self.locals['rewards'][0]
        
        # Track distance
        info = self.locals['infos'][0]
        if 'distance' in info:
            self.current_episode_distances.append(info['distance'])
        
        # Check if episode ended
        done = self.locals['dones'][0]
        if done:
            # Store episode metrics
            self.episode_rewards.append(self.current_episode_reward)
            
            if self.current_episode_distances:
                avg_distance = sum(self.current_episode_distances) / len(self.current_episode_distances)
                final_distance = self.current_episode_distances[-1]
                self.episode_distances.append({
                    'avg_distance': avg_distance,
                    'final_distance': final_distance,
                    'initial_distance': self.current_episode_distances[0]
                })
            
            # Print episode summary
            ep_num = len(self.episode_rewards)
            print(f"\n{'='*60}")
            print(f"Episode {ep_num} completed:")
            print(f"  - Total reward: {self.current_episode_reward:.2f}")
            if self.current_episode_distances:
                print(f"  - Initial distance: {self.current_episode_distances[0]:.1f}")
                print(f"  - Final distance: {self.current_episode_distances[-1]:.1f}")
                print(f"  - Average distance: {avg_distance:.1f}")
            print(f"{'='*60}\n")
            
            # Reset episode counters
            self.current_episode_reward = 0
            self.current_episode_distances = []
        
        return True
    
    def get_metrics_df(self):
        """
        Get episode metrics as pandas DataFrame.
        
        Returns:
            pd.DataFrame: Episode metrics
        """
        data = {
            'episode': list(range(1, len(self.episode_rewards) + 1)),
            'total_reward': self.episode_rewards,
        }
        
        if self.episode_distances:
            data['avg_distance'] = [d['avg_distance'] for d in self.episode_distances]
            data['final_distance'] = [d['final_distance'] for d in self.episode_distances]
            data['initial_distance'] = [d['initial_distance'] for d in self.episode_distances]
        
        return pd.DataFrame(data)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_episode_metrics(metrics_df, save_suffix="", title_prefix=""):
    """
    Generate episode metrics plots.
    
    Creates a 2x2 grid of plots showing:
        - Total reward per episode
        - Distance improvement per episode
        - Distance evolution (initial, final, average)
        
    Args:
        metrics_df: DataFrame with episode metrics
        save_suffix: Suffix for output filename
        title_prefix: Prefix for plot title
    """
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f'{title_prefix} - Training Metrics' if title_prefix else 'Training Metrics per Episode'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Total reward per episode
    ax1 = axes[0, 0]
    ax1.plot(metrics_df['episode'], metrics_df['total_reward'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Total Reward', fontweight='bold')
    ax1.set_title('Total Reward per Episode')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance improvement per episode
    ax2 = axes[0, 1]
    if 'initial_distance' in metrics_df.columns and 'final_distance' in metrics_df.columns:
        improvement = metrics_df['initial_distance'] - metrics_df['final_distance']
        colors = ['green' if x > 0 else 'red' for x in improvement]
        ax2.bar(metrics_df['episode'], improvement, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Distance Improvement', fontweight='bold')
        ax2.set_title('Distance Improvement (Initial - Final)')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No improvement data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distance Improvement')
    
    # Plot 3: Distance evolution
    ax3 = axes[1, 0]
    if 'avg_distance' in metrics_df.columns:
        ax3.plot(metrics_df['episode'], metrics_df['initial_distance'], 
                marker='s', label='Initial Distance', linewidth=2, markersize=6, color='blue')
        ax3.plot(metrics_df['episode'], metrics_df['final_distance'], 
                marker='o', label='Final Distance', linewidth=2, markersize=6, color='red')
        ax3.plot(metrics_df['episode'], metrics_df['avg_distance'], 
                marker='^', label='Average Distance', linewidth=2, markersize=6, color='green')
        ax3.set_xlabel('Episode', fontweight='bold')
        ax3.set_ylabel('Distance', fontweight='bold')
        ax3.set_title('Distance Evolution to Target')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No distance data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Distance Evolution')
    
    # Hide bottom-right subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, f"episode_metrics{save_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nMetrics plot saved at: {plot_path}")
    plt.close()


# ============================================================================
# CURRICULUM LEARNING TRAINING
# ============================================================================

def train_model_curriculum():
    """
    Train model using curriculum learning approach.
    
    Two-phase training:
        1. Phase 1: Static target (easier task)
        2. Phase 2: Moving target (harder task, builds on phase 1)
    
    Returns:
        model: Trained SAC model
        metrics_phase1: Phase 1 metrics DataFrame
        metrics_phase2: Phase 2 metrics DataFrame
    """
    print(f"\n{'='*70}")
    print(f"STARTING CURRICULUM LEARNING TRAINING")
    print(f"{'='*70}\n")

    # ========================================================================
    # PHASE 1: Static Target
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"PHASE 1: Training with Static Target (CYLINDER DOES NOT MOVE)")
    print(f"{'='*70}\n")
    
    def make_env_static():
        """Create environment with static target."""
        monitor_filename = os.path.join(LOG_DIR, "phase1_monitor.csv")
        env = CustomEnv(size=1000, max_steps=MAX_STEPS_PER_EPISODE)
        env.target_move_frequency = 999  # Target won't move
        env = Monitor(env, filename=monitor_filename)
        return env
    
    env_phase1 = DummyVecEnv([make_env_static])
    
    # Initialize SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env_phase1,
        learning_rate=3e-4,
        gamma=0.95,
        buffer_size=2000,
        learning_starts=500,
        batch_size=64,
        train_freq=2,
        gradient_steps=2,
        tau=0.005,
        target_update_interval=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "phase1")
    )

    # Train phase 1
    metrics_callback_phase1 = EpisodeMetricsCallback(verbose=1)
    model.learn(
        total_timesteps=TIMESTEPS_PHASE1, 
        callback=metrics_callback_phase1, 
        progress_bar=True
    )

    # Save phase 1 model
    phase1_model_path = os.path.join(MODELS_DIR, "sac_cylinder_phase1")
    model.save(phase1_model_path)
    print(f"\nPhase 1 model saved at: {phase1_model_path}")

    # Save phase 1 metrics
    metrics_df_phase1 = metrics_callback_phase1.get_metrics_df()
    metrics_df_phase1['phase'] = 1
    metrics_path_phase1 = os.path.join(LOG_DIR, "phase1_metrics.csv")
    metrics_df_phase1.to_csv(metrics_path_phase1, index=False)
    print(f"Phase 1 metrics saved at: {metrics_path_phase1}")
    
    plot_episode_metrics(metrics_df_phase1, save_suffix="_phase1", 
                        title_prefix="Phase 1 (Static Target)")

    # ========================================================================
    # PHASE 2: Moving Target
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"PHASE 2: Training with Moving Target (NOW IT MOVES)")
    print(f"{'='*70}\n")
    
    def make_env_moving():
        """Create environment with moving target."""
        monitor_filename = os.path.join(LOG_DIR, "phase2_monitor.csv")
        env = CustomEnv(size=1000, max_steps=MAX_STEPS_PER_EPISODE)
        env.target_move_frequency = 1  # Target moves every step
        env = Monitor(env, filename=monitor_filename)
        return env

    env_phase2 = DummyVecEnv([make_env_moving])
    model.set_env(env_phase2)

    # Train phase 2 (continue from phase 1)
    metrics_callback_phase2 = EpisodeMetricsCallback(verbose=1)
    model.learn(
        total_timesteps=TIMESTEPS_PHASE2, 
        callback=metrics_callback_phase2, 
        progress_bar=True, 
        reset_num_timesteps=False  # Continue from phase 1
    )

    # Save final model
    final_model_path = os.path.join(MODELS_DIR, "sac_cylinder_final")
    model.save(final_model_path)
    print(f"\nFinal model saved at: {final_model_path}")

    # Save phase 2 metrics
    metrics_df_phase2 = metrics_callback_phase2.get_metrics_df()
    metrics_df_phase2['phase'] = 2
    metrics_path_phase2 = os.path.join(LOG_DIR, "phase2_metrics.csv")
    metrics_df_phase2.to_csv(metrics_path_phase2, index=False)
    print(f"Phase 2 metrics saved at: {metrics_path_phase2}")
    
    plot_episode_metrics(metrics_df_phase2, save_suffix="_phase2", 
                        title_prefix="Phase 2 (Moving Target)")

    # ========================================================================
    # TRAINING SUMMARY
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"✓ TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Phase 1 episodes: {len(metrics_df_phase1)}")
    print(f"Phase 2 episodes: {len(metrics_df_phase2)}")
    print(f"Total episodes: {len(metrics_df_phase1) + len(metrics_df_phase2)}")
    print(f"{'='*70}\n")

    return model, metrics_df_phase1, metrics_df_phase2

if __name__ == "__main__":
    model, metrics_phase1, metrics_phase2 = train_model_curriculum()
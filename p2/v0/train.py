import os
import pandas as pd

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import CustomEnv


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

MAX_STEPS_PER_EPISODE = 50
TOTAL_TIMESTEPS = 50000
TOTAL_EPISODES = TOTAL_TIMESTEPS // MAX_STEPS_PER_EPISODE

print(f"\n{'=' * 70}")
print("TRAINING CONFIGURATION (STATIC TARGET)")
print(f"{'=' * 70}")
print(f"Steps per episode: {MAX_STEPS_PER_EPISODE}")
print(f"Total Episodes: ~{TOTAL_EPISODES}")
print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
print(f"{'=' * 70}\n")


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
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_distances = []
        self.current_episode_reward = 0
        self.current_episode_distances = []

    def _on_step(self):
        """Called at each step of the environment."""
        self.current_episode_reward += self.locals["rewards"][0]
        info = self.locals["infos"][0]
        if "distance" in info:
            self.current_episode_distances.append(info["distance"])

        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            if self.current_episode_distances:
                avg_distance = sum(self.current_episode_distances) / len(
                    self.current_episode_distances
                )
                final_distance = self.current_episode_distances[-1]
                self.episode_distances.append(
                    {
                        "avg_distance": avg_distance,
                        "final_distance": final_distance,
                        "initial_distance": self.current_episode_distances[0],
                    }
                )

            ep_num = len(self.episode_rewards)
            print(f"\n{'=' * 60}")
            print(f"Episode {ep_num} completed:")
            print(f"  - Total reward: {self.current_episode_reward:.2f}")
            if self.current_episode_distances:
                print(f"  - Initial distance: {self.current_episode_distances[0]:.1f}")
                print(f"  - Final distance: {self.current_episode_distances[-1]:.1f}")
                print(f"  - Average distance: {avg_distance:.1f}")
            print(f"{'=' * 60}\n")

            self.current_episode_reward = 0
            self.current_episode_distances = []
        return True

    def get_metrics_df(self):
        """Get episode metrics as pandas DataFrame."""
        data = {
            "episode": list(range(1, len(self.episode_rewards) + 1)),
            "total_reward": self.episode_rewards,
        }
        if self.episode_distances:
            data["avg_distance"] = [d["avg_distance"] for d in self.episode_distances]
            data["final_distance"] = [
                d["final_distance"] for d in self.episode_distances
            ]
            data["initial_distance"] = [
                d["initial_distance"] for d in self.episode_distances
            ]
        return pd.DataFrame(data)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_model():
    """
    Train a SAC model with a static target.
    """
    print(f"\n{'=' * 70}")
    print("STARTING TRAINING: STATIC TARGET")
    print(f"{'=' * 70}\n")

    def make_env():
        """Create environment with a static target."""
        monitor_filename = os.path.join(LOG_DIR, "monitor.csv")
        env = CustomEnv(size=1000, max_steps=MAX_STEPS_PER_EPISODE)
        env.target_move_frequency = 999  # Target does not move
        env = Monitor(env, filename=monitor_filename)
        return env

    env = DummyVecEnv([make_env])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.95,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        train_freq=(1, "step"),
        gradient_steps=-1,
        tau=0.005,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    metrics_callback = EpisodeMetricsCallback(verbose=1)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, callback=metrics_callback, progress_bar=True
    )

    final_model_path = os.path.join(MODELS_DIR, "sac_cylinder_final")
    model.save(final_model_path)
    print(f"\nFinal model saved at: {final_model_path}.zip")

    metrics_df = metrics_callback.get_metrics_df()
    metrics_path = os.path.join(LOG_DIR, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Training metrics saved at: {metrics_path}")

    print(f"\n{'=' * 70}")
    print("✓ TRAINING COMPLETED")
    print(f"Total episodes: {len(metrics_df)}")
    print(f"{'=' * 70}\n")

    return model, metrics_df


if __name__ == "__main__":
    train_model()

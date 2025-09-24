import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from cylinder_env import CylinderEnv

TOTAL_STEPS = 3
TRAIN_STEPS = 3
N_STEPS = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FINAL_PATH = os.path.join(MODELS_DIR, "ppo_cylinder")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def make_env():
    e = CylinderEnv()
    e = Monitor(e, filename=os.path.join(LOG_DIR, "train_monitor.csv"))
    return e

env = DummyVecEnv([make_env])


def train(model):

    metrics = []
    for _ in range(TRAIN_STEPS):
        model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_callback, ckpt_callback])
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2, deterministic=True)
        metrics.append((mean_reward, std_reward, model))
        print(f"Eval reward: mean={mean_reward:.2f} ± {std_reward:.2f}")

    model = max(metrics, key=lambda x: x[0])[2]
    model.save(FINAL_PATH)
    print(f"Modelo guardado en: {FINAL_PATH}.zip")

def plot_training_rewards():
    """Plot training rewards from the Monitor CSV file"""
    # Gráfica de recompensa (rolling) para la memoria
    csv_path = os.path.join(LOG_DIR, "train_monitor.csv")
    df = pd.read_csv(csv_path, comment="#")
    df["rolling_ep_rew_mean"] = df["r"].rolling(window=20, min_periods=1).mean()
    sns.set_context("talk")
    plt.figure()
    df.reset_index()
    sns.lineplot(data=df, x='t', y="rolling_ep_rew_mean")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa media (rolling 20)")
    plt.title("Evolución de recompensa (PPO)")
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, "ppo_training_rewards.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Gráfica guardada: {plot_path}")

if __name__ == "__main__":
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=N_STEPS,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        verbose=1,
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=CKPT_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    ckpt_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CKPT_DIR,
        name_prefix="ppo_cylinder",
    )
    train(model)
    plot_training_rewards()
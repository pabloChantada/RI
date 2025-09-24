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

LOG_DIR = "logs/ppo"
BEST_DIR = "models/ppo_best"
CKPT_DIR = "checkpoints/ppo"
TB_DIR = "runs/ppo"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

def make_env():
    env = CylinderEnv()
    return Monitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))

vec_env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    learning_rate=3e-4,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    verbose=1,
    tensorboard_log=TB_DIR,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_DIR,
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

TOTAL_STEPS = 200_000
model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_callback, ckpt_callback])

FINAL_PATH = "models/ppo_cylinder_final"
os.makedirs("models", exist_ok=True)
model.save(FINAL_PATH)  # → crea models/ppo_cylinder_final.zip
print(f"Modelo guardado en: {FINAL_PATH}.zip")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Eval reward: mean={mean_reward:.2f} ± {std_reward:.2f}")

# Gráfica de recompensa (rolling) para la memoria
csv_path = os.path.join(LOG_DIR, "monitor.csv")
df = pd.read_csv(csv_path, comment="#")
df["rolling_ep_rew_mean"] = df["r"].rolling(window=20, min_periods=1).mean()
sns.set_context("talk")
plt.figure()
sns.lineplot(data=df, x=df.index, y="rolling_ep_rew_mean")
plt.xlabel("Episodios")
plt.ylabel("Recompensa media (rolling 20)")
plt.title("Evolución de recompensa (PPO)")
plt.tight_layout()
plt.savefig("ppo_training_rewards.png", dpi=150)
print("Gráfica guardada: ppo_training_rewards.png")

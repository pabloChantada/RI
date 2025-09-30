import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from cylinder_env import CylinderEnv

# ===================== Config =====================
TRAIN_STEPS = 10       # pasos por "episodio de entrenamiento" (ajusta a lo que quieras)
EPISODES = 3           # repeticiones del bucle de train + evaluate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FINAL_PATH = os.path.join(MODELS_DIR, "ppo_cylinder")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Alterna Objetivo 1 (False) u Objetivo 2 (True) desde aquí:
ENV_KWARGS = dict(
    max_steps=TRAIN_STEPS,
    move_target=False,          # False: acercarse (Obj.1). True: seguir en movimiento (Obj.2)
)

def make_env():
    monitor_filename = os.path.join(LOG_DIR, "train_monitor.csv")
    e = CylinderEnv(**ENV_KWARGS)
    e = Monitor(e, filename=monitor_filename)
    return e

# VecEnv
env = DummyVecEnv([make_env])

# ===================== Modelo =====================
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=TRAIN_STEPS,
    batch_size=128,
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
    eval_freq=10_000,          # con tus TRAIN_STEPS bajos, probablemente no dispare; no es obligatorio
    n_eval_episodes=5,
    deterministic=True,
)

ckpt_callback = CheckpointCallback(
    save_freq=10_000,          # idem arriba
    save_path=CKPT_DIR,
    name_prefix="ppo_cylinder",
)

# ===================== Entrenamiento =====================
def train(model):
    metrics = []
    for i in range(EPISODES):
        print(f"--- Entrenamiento: {i+1}/{EPISODES} ---")
        env.reset()
        model.learn(total_timesteps=TRAIN_STEPS, callback=[eval_callback, ckpt_callback])
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2, deterministic=True)
        metrics.append({"step": (i+1)*TRAIN_STEPS, "mean_reward": mean_reward, "std_reward": std_reward})
        print(f"Eval reward: mean={mean_reward:.2f} ± {std_reward:.2f}")

    # Guarda métricas (para el requisito del PDF de graficar mean_reward/ep_rew_mean)
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(LOG_DIR, "ppo_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Métricas guardadas en: {metrics_path}")

    # Guarda el modelo final
    best_idx = metrics_df["mean_reward"].idxmax()
    print(f"Mejor recompensa media: {metrics_df.loc[best_idx, 'mean_reward']:.2f}")
    model.save(FINAL_PATH)
    print(f"Modelo guardado en: {FINAL_PATH}.zip")

# ===================== Gráficas requeridas por el PDF =====================
def plot_training_rewards():
    """
    Gráfica tipo 'ep_rew_mean' (aproximada) a partir del Monitor CSV.
    El Monitor guarda 'r' (recompensa por episodio); aquí mostramos un rolling mean.
    """
    csv_path = os.path.join(LOG_DIR, "train_monitor.csv")
    if not os.path.exists(csv_path):
        print("No se encontró train_monitor.csv. ¿Llegó a ejecutarse al menos un episodio?")
        return

    df = pd.read_csv(csv_path, comment="#")
    # 'r' es la recompensa por episodio; creamos un índice de episodios explícito
    df["episode"] = np.arange(1, len(df) + 1)
    # Rolling (ventana 10) como proxy de 'ep_rew_mean'
    df["ep_rew_mean"] = df["r"].rolling(window=10, min_periods=1).mean()

    sns.set_context("talk")
    plt.figure()
    ax = sns.lineplot(data=df, x="episode", y="ep_rew_mean")
    plt.xlabel("Episodio")
    plt.ylabel("ep_rew_mean (rolling=10)")
    plt.title("Evolución de ep_rew_mean (Monitor)")
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, "ppo_ep_rew_mean.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Gráfica guardada: {plot_path}")

def plot_mean_reward_from_evaluate():
    """
    Gráfica de 'mean_reward' (evaluate_policy) desde ppo_metrics.csv.
    Esto responde literalmente al requisito del PDF de representar 'mean_reward' en PNG.
    """
    csv_path = os.path.join(LOG_DIR, "ppo_metrics.csv")
    if not os.path.exists(csv_path):
        print("No se encontró ppo_metrics.csv. ¿Se ejecutó train(model)?")
        return

    df = pd.read_csv(csv_path)
    sns.set_context("talk")
    plt.figure()
    ax = sns.lineplot(data=df, x="step", y="mean_reward", marker="o")
    if "std_reward" in df.columns:
        plt.fill_between(df["step"], df["mean_reward"] - df["std_reward"],
                         df["mean_reward"] + df["std_reward"], alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward (evaluate_policy)")
    plt.title("PPO Mean Reward vs Timesteps")
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, "ppo_mean_reward.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Gráfica guardada: {plot_path}")

if __name__ == "__main__":
    train(model)
    # Requisitos de “Representación de resultados” (PNG, seaborn). :contentReference[oaicite:2]{index=2}
    plot_training_rewards()
    plot_mean_reward_from_evaluate()

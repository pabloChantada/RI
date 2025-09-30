import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from cylinder_env import CylinderEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def pos(info):
    return info["agent_position"], info["target_position"]

def run_evaluation(env, model, max_steps=300, deterministic=True):
    obs, info = env.reset()
    done = False
    traj_agent, traj_target = [], []

    a0, t0 = pos(info)
    traj_agent.append(a0); traj_target.append(t0)

    step = 0
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        a, t = pos(info)
        traj_agent.append(a); traj_target.append(t)
        done = bool(terminated or truncated)
        step += 1

    return traj_agent, traj_target

def plot_trajectories(traj_agent, traj_target, half, out_name):
    """Plot 2D trajectories of agent and target with dynamic limits"""
    fig, ax = plt.subplots()
    lim = float(half)
    ax.set_xlim(-lim, +lim)
    ax.set_ylim(-lim, +lim)
    ax.set_aspect("equal", adjustable="box")

    ax.plot([p[0] for p in traj_agent],  [p[1] for p in traj_agent],  label="Agente")
    ax.plot([p[0] for p in traj_target], [p[1] for p in traj_target], label="Cilindro", linestyle="--")
    ax.scatter(traj_agent[0][0],  traj_agent[0][1],  marker="o", label="Inicio agente")
    ax.scatter(traj_target[0][0], traj_target[0][1], marker="x", label="Inicio cilindro")
    ax.scatter(traj_agent[-1][0], traj_agent[-1][1], marker="*", color="red", label="Fin agente")

    ax.grid(True); ax.legend()
    plt.xlabel("X"); plt.ylabel("Z"); plt.title("Trayectorias 2D")
    plt.tight_layout()
    out_path = os.path.join(LOGS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Gráfica guardada: {out_path}")

if __name__ == "__main__":
    model_path = os.path.join(MODELS_DIR, "ppo_cylinder.zip")

    # ===== Evaluación Objetivo 1: cilindro estático =====
    env_static = CylinderEnv(move_target=False)
    model = PPO.load(model_path, env=env_static, device="auto")
    traj_agent, traj_target = run_evaluation(env_static, model)
    plot_trajectories(traj_agent, traj_target, env_static.half, "trayectorias_2D_static.png")

    # ===== Evaluación Objetivo 2: cilindro en movimiento =====
    env_moving = CylinderEnv(move_target=True)
    model.set_env(env_moving)  # reutilizamos el mismo modelo con otro entorno compatible
    traj_agent_m, traj_target_m = run_evaluation(env_moving, model)
    plot_trajectories(traj_agent_m, traj_target_m, env_moving.half, "trayectorias_2D_moving.png")

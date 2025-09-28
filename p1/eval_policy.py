import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from cylinder_env import CylinderEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

def pos(info_list):
    return info_list["agent_position"], info_list["target_position"]

def run_evaluation(env, model):
    obs, info = env.reset()
    done = False
    traj_agent, traj_target = [], []

    a0, t0 = pos(info)
    traj_agent.append(a0); traj_target.append(t0)

    step, MAX_STEPS = 0, 300
    while not done and step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        a, t = pos(info)
        traj_agent.append(a); traj_target.append(t)
        done = bool(terminated or truncated)
        step += 1

    return traj_agent, traj_target

def plot_trajectories(traj_agent, traj_target):
    """Plot 2D trajectories of agent and target"""
    ax = plt.figure().gca()
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_aspect("equal", adjustable="box")
    ax.plot([p[0] for p in traj_agent], [p[1] for p in traj_agent], label="Agente")
    ax.plot([p[0] for p in traj_target], [p[1] for p in traj_target], label="Cilindro", linestyle="--")
    ax.scatter(traj_agent[0][0], traj_agent[0][1], marker="o", label="Inicio agente")
    ax.scatter(traj_target[0][0], traj_target[0][1], marker="x", label="Inicio cilindro")
    ax.scatter(traj_agent[-1][0], traj_agent[-1][1], marker="*", color="red", label="Fin agente")
    ax.grid(True); ax.legend()
    plt.xlabel("X"); plt.ylabel("Z"); plt.title("Trayectorias 2D")
    os.makedirs(LOGS_DIR, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(LOGS_DIR, "trayectorias_2D.png"), dpi=150)
    print("Gráfica guardada: logs/trayectorias_2D.png")

if __name__ == "__main__":

    env = CylinderEnv() 
    model = PPO.load(os.path.join(MODELS_DIR, "ppo_cylinder.zip"), env=env, device="auto")

    traj_agent, traj_target = run_evaluation(env, model)
    plot_trajectories(traj_agent, traj_target)
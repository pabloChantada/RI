import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from cylinder_env import CylinderEnv

MODEL_PATH = "models/ppo_cylinder_final.zip"  # o models/ppo_best/best_model.zip

env = CylinderEnv() 
model = PPO.load(MODEL_PATH, env=env, device="auto")

obs, info = env.reset()
done = False
traj_agent, traj_target = [], []

def pos(info_list):
    return info_list["agent_position"], info_list["target_position"]

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




ax = plt.figure().gca()
ax.set_aspect("equal", adjustable="box")
ax.plot([p[0] for p in traj_agent], [p[1] for p in traj_agent], label="Agente")
ax.plot([p[0] for p in traj_target], [p[1] for p in traj_target], label="Cilindro", linestyle="--")
ax.scatter(traj_agent[0][0], traj_agent[0][1], marker="o", label="Inicio agente")
ax.scatter(traj_target[0][0], traj_target[0][1], marker="x", label="Inicio cilindro")
ax.grid(True); ax.legend()
plt.xlabel("X"); plt.ylabel("Z"); plt.title("Trayectorias 2D")
os.makedirs("logs", exist_ok=True)
plt.tight_layout(); plt.savefig("logs/trayectorias_2D.png", dpi=150)
print("Gráfica guardada: logs/trayectorias_2D.png")

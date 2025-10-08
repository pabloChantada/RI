import os
import numpy as np
import matplotlib.pyplot as plt
from env import CustomEnv
from stable_baselines3 import SAC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(LOGS_DIR, exist_ok=True)

def pos(info):
    return info["agent_position"], info["target_position"]

def run_evaluation(env, model, max_steps=30):
    """
    Ejecuta una evaluación completa del modelo
    
    Args:
        env: Entorno de Gymnasium
        model: Modelo entrenado
        max_steps: Máximo número de pasos (debe coincidir con max_steps del env)
    
    Returns:
        traj_agent: Lista de posiciones del agente
        traj_target: Lista de posiciones del objetivo
        episode_reward: Recompensa total del episodio
        episode_info: Información adicional del episodio
    """
    obs, info = env.reset()
    done = False
    traj_agent, traj_target = [], []
    episode_reward = 0
    distances = []

    # Valores iniciales 
    agent0, target0 = pos(info)
    traj_agent.append(agent0)
    traj_target.append(target0)
    distances.append(info['distance'])

    step = 0
    while not done and step < max_steps:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Valores de episodio
        episode_reward += reward
        agent, target = pos(info)
        traj_agent.append(agent)
        traj_target.append(target)
        distances.append(info['distance'])
        
        # Comprobar terminacion temprana
        done = bool(terminated or truncated)
        step += 1
    
    episode_info = {
        'final_distance': distances[-1],
        'initial_distance': distances[0],
        'avg_distance': np.mean(distances),
        'min_distance': np.min(distances),
        'success': distances[-1] < 150  # Objetivo alcanzado
    }
    
    plot_trajectories(traj_agent, traj_target, episode_info, 
                             out_name="evaluation_trajectory.png")

    # return traj_agent, traj_target, episode_reward, episode_info

def plot_trajectories(traj_agent, traj_target, episode_info, half=1000, out_name="trajectories_2d.png"):
    """
    Plot 2D trajectories of agent and target with enhanced visualization
    
    Args:
        traj_agent: Lista de posiciones del agente [(x, z), ...]
        traj_target: Lista de posiciones del objetivo [(x, z), ...]
        episode_info: Diccionario con información del episodio
        half: Mitad del tamaño del entorno para límites
        out_name: Nombre del archivo de salida
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    lim = float(half)
    ax.set_xlim(-lim, +lim)
    ax.set_ylim(-lim, +lim)
    ax.set_aspect("equal", adjustable="box")

    # Trayectoria del agente
    agent_x = [p[0] for p in traj_agent]
    agent_z = [p[1] for p in traj_agent]

    # Plot sin coloreado por tiempo
    ax.plot(agent_x, agent_z, 'b-', alpha=0.6, linewidth=2, label='Trayectoria agente')
    
    # Trayectoria del objetivo
    target_x = [p[0] for p in traj_target]
    target_z = [p[1] for p in traj_target]
    ax.plot(target_x, target_z, 'r--', alpha=0.5, linewidth=2, label='Trayectoria cilindro')
    
    # Marcadores especiales
    ax.scatter(agent_x[0], agent_z[0], marker='x', s=200, 
              color='green', edgecolors='black', linewidths=2,
              label='Inicio agente', zorder=5)
    ax.scatter(target_x[0], target_z[0], marker='o', s=200,
              color='red', linewidths=3,
              label='Inicio cilindro', zorder=5)
    ax.scatter(agent_x[-1], agent_z[-1], marker='*', s=300,
              color='blue', linewidths=2,
              label='Fin agente', zorder=5)
    
    # Grid y leyenda
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Título con información del episodio
    title = f"Trayectorias 2D - Evaluación\n"
    title += f"Distancia final: {episode_info['final_distance']:.1f} | "
    title += f"Éxito: {'Sí' if episode_info['success'] else 'No'}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.xlabel("X", fontweight='bold')
    plt.ylabel("Z", fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(LOGS_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada: {out_path}")
    plt.close()


# def run_multiple_evaluations(env, model, num_evals=5):
#     """
#     Ejecuta múltiples evaluaciones y muestra estadísticas
#     """
#     print(f"\n{'='*60}")
#     print(f"EJECUTANDO {num_evals} EVALUACIONES")
#     print(f"{'='*60}\n")
#
#     all_rewards = []
#     all_distances = []
#     successes = 0
#
#     for i in range(num_evals):
#         print(f"Evaluación {i+1}/{num_evals}...")
#         traj_agent, traj_target, reward, info = run_evaluation(
#             env, model
#         )
#
#         all_rewards.append(reward)
#         all_distances.append(info['final_distance'])
#         if info['success']:
#             successes += 1
#
#         print(f"  Recompensa: {reward:.2f}")
#         print(f"  Distancia final: {info['final_distance']:.1f}")
#         print(f"  Éxito: {'Sí' if info['success'] else 'No'}\n")
#
#         # Guardar trayectoria de la primera evaluación
#         if i == 0:
#             plot_trajectories(traj_agent, traj_target, info, 
#                             out_name="evaluation_trajectory.png")
#
#     # Estadísticas finales
#     print(f"\n{'='*60}")
#     print(f"ESTADÍSTICAS DE EVALUACIÓN")
#     print(f"{'='*60}")
#     print(f"Recompensa promedio: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
#     print(f"Distancia final promedio: {np.mean(all_distances):.1f} ± {np.std(all_distances):.1f}")
#     print(f"Tasa de éxito: {successes}/{num_evals} ({100*successes/num_evals:.1f}%)")
#     print(f"{'='*60}\n")
#


if __name__ == "__main__":
    # Crear entorno con max_steps consistente
    env = CustomEnv(size=1000, max_steps=30)
    
    # Cargar modelo
    model_path = os.path.join(MODELS_DIR, "sac_cylinder_final.zip")
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo en {model_path}")
        print("Asegúrate de entrenar el modelo primero con train_ppo.py")
        exit(1)
    
    print(f"Cargando modelo desde: {model_path}")
    model = SAC.load(model_path, env=env, device="cpu")
    
    run_evaluation(env, model)
    # Ejecutar evaluaciones
    # run_multiple_evaluations(env, model, num_evals=5)
    
    # Cerrar entorno
    env.close()


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
# CONFIGURACIÓN DE ENTRENAMIENTO (2 FASES)
# ============================================================================

MAX_STEPS_PER_EPISODE = 30   # Pasos máximos por episodio

# FASE 1: Target Estático
EPISODES_PHASE1 = 234
TIMESTEPS_PHASE1 = MAX_STEPS_PER_EPISODE * EPISODES_PHASE1  # 6000 timesteps

# FASE 2: Target Móvil
EPISODES_PHASE2 = 100
TIMESTEPS_PHASE2 = MAX_STEPS_PER_EPISODE * EPISODES_PHASE2  # 3000 timesteps

# Total del entrenamiento
TOTAL_EPISODES = EPISODES_PHASE1 + EPISODES_PHASE2  # 300 episodios
TOTAL_TIMESTEPS = TIMESTEPS_PHASE1 + TIMESTEPS_PHASE2  # 9000 timesteps

print(f"\n{'='*70}")
print(f"CONFIGURACIÓN DE ENTRENAMIENTO")
print(f"{'='*70}")
print(f"Pasos por episodio: {MAX_STEPS_PER_EPISODE}")
print(f"\nFASE 1 (Target Estático - NO SE MUEVE):")
print(f"  - Episodios: {EPISODES_PHASE1}")
print(f"  - Timesteps: {TIMESTEPS_PHASE1}")
print(f"\nFASE 2 (Target Móvil - SE MUEVE CADA 10 PASOS):")
print(f"  - Episodios: {EPISODES_PHASE2}")
print(f"  - Timesteps: {TIMESTEPS_PHASE2}")
print(f"\nTOTAL:")
print(f"  - Episodios: {TOTAL_EPISODES}")
print(f"  - Timesteps: {TOTAL_TIMESTEPS}")
print(f"{'='*70}\n")

# ============================================================================
# DIRECTORIOS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# CALLBACK PARA MÉTRICAS
# ============================================================================

class EpisodeMetricsCallback(BaseCallback):
    """Callback personalizado para registrar métricas por episodio"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_distances = []
        self.current_episode_reward = 0
        self.current_episode_distances = []
        
    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]
        
        info = self.locals['infos'][0]
        if 'distance' in info:
            self.current_episode_distances.append(info['distance'])
        
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            
            if self.current_episode_distances:
                avg_distance = sum(self.current_episode_distances) / len(self.current_episode_distances)
                final_distance = self.current_episode_distances[-1]
                self.episode_distances.append({
                    'avg_distance': avg_distance,
                    'final_distance': final_distance,
                    'initial_distance': self.current_episode_distances[0]
                })
            
            ep_num = len(self.episode_rewards)
            print(f"\n{'='*60}")
            print(f"Episodio {ep_num} completado:")
            print(f"  - Recompensa total: {self.current_episode_reward:.2f}")
            if self.current_episode_distances:
                print(f"  - Distancia inicial: {self.current_episode_distances[0]:.1f}")
                print(f"  - Distancia final: {self.current_episode_distances[-1]:.1f}")
                print(f"  - Distancia promedio: {avg_distance:.1f}")
            print(f"{'='*60}\n")
            
            self.current_episode_reward = 0
            self.current_episode_distances = []
        
        return True
    
    def get_metrics_df(self):
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
# FUNCIONES AUXILIARES
# ============================================================================

def plot_episode_metrics(metrics_df, save_suffix="", title_prefix=""):
    """Genera gráficas de métricas por episodio"""
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f'{title_prefix} - Métricas de Entrenamiento' if title_prefix else 'Métricas de Entrenamiento por Episodio'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Recompensa total por episodio (top-left)
    ax1 = axes[0, 0]
    ax1.plot(metrics_df['episode'], metrics_df['total_reward'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Episodio', fontweight='bold')
    ax1.set_ylabel('Recompensa Total', fontweight='bold')
    ax1.set_title('Recompensa Total por Episodio')
    ax1.grid(True, alpha=0.3)
    
    # Mejora en distancia por episodio (top-right)
    ax2 = axes[0, 1]
    if 'initial_distance' in metrics_df.columns and 'final_distance' in metrics_df.columns:
        improvement = metrics_df['initial_distance'] - metrics_df['final_distance']
        colors = ['green' if x > 0 else 'red' for x in improvement]
        ax2.bar(metrics_df['episode'], improvement, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Episodio', fontweight='bold')
        ax2.set_ylabel('Mejora en Distancia', fontweight='bold')
        ax2.set_title('Mejora en Distancia (Inicial - Final)')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No hay datos de mejora', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Mejora en Distancia')
    
    # Distancia al objetivo (bottom-left)
    ax3 = axes[1, 0]
    if 'avg_distance' in metrics_df.columns:
        ax3.plot(metrics_df['episode'], metrics_df['initial_distance'], 
                marker='s', label='Distancia Inicial', linewidth=2, markersize=6, color='blue')
        ax3.plot(metrics_df['episode'], metrics_df['final_distance'], 
                marker='o', label='Distancia Final', linewidth=2, markersize=6, color='red')
        ax3.plot(metrics_df['episode'], metrics_df['avg_distance'], 
                marker='^', label='Distancia Promedio', linewidth=2, markersize=6, color='green')
        ax3.set_xlabel('Episodio', fontweight='bold')
        ax3.set_ylabel('Distancia', fontweight='bold')
        ax3.set_title('Evolución de la Distancia al Objetivo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No hay datos de distancia', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Evolución de la Distancia')
    
    # Ocultar el último subplot (bottom-right)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, f"episode_metrics{save_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n Gráfica de métricas guardada en: {plot_path}")
    plt.close()

# ============================================================================
# ENTRENAMIENTO CON CURRICULUM LEARNING
# ============================================================================

def train_model_curriculum():
    """Entrenamiento en dos fases: Fase 1 (Target estático) y Fase 2 (Target móvil)"""
    print(f"\n{'='*70}")
    print(f"🎓 INICIANDO ENTRENAMIENTO CON CURRICULUM LEARNING")
    print(f"{'='*70}\n")

    # ========================================================================
    # FASE 1: Target Estático
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"FASE 1: Entrenamiento con Target Estático (EL CILINDRO NO SE MUEVE)")
    print(f"{'='*70}\n")
    
    def make_env_static():
        monitor_filename = os.path.join(LOG_DIR, "phase1_monitor.csv")
        env = CustomEnv(size=1000, max_steps=MAX_STEPS_PER_EPISODE)
        env.target_move_frequency = 999  # Target estático
        env = Monitor(env, filename=monitor_filename)
        return env
    
    env_phase1 = DummyVecEnv([make_env_static])
    
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

    metrics_callback_phase1 = EpisodeMetricsCallback(verbose=1)
    model.learn(
        total_timesteps=TIMESTEPS_PHASE1, 
        callback=metrics_callback_phase1, 
        progress_bar=True
    )

    # Guardar modelo de fase 1
    phase1_model_path = os.path.join(MODELS_DIR, "sac_cylinder_phase1")
    model.save(phase1_model_path)
    print(f"\n Modelo Fase 1 guardado en: {phase1_model_path}")

    # Guardar métricas de fase 1
    metrics_df_phase1 = metrics_callback_phase1.get_metrics_df()
    metrics_df_phase1['phase'] = 1
    metrics_path_phase1 = os.path.join(LOG_DIR, "phase1_metrics.csv")
    metrics_df_phase1.to_csv(metrics_path_phase1, index=False)
    print(f"Métricas Fase 1 guardadas en: {metrics_path_phase1}")
    
    plot_episode_metrics(metrics_df_phase1, save_suffix="_phase1", title_prefix="Fase 1 (Target Estático)")

    # ========================================================================
    # FASE 2: Target Móvil
    # ========================================================================

    print(f"\n{'='*70}")
    print(f" FASE 2: Entrenamiento con Target Móvil (AHORA SÍ SE MUEVE)")
    print(f"{'='*70}\n")
    
    def make_env_moving():
        monitor_filename = os.path.join(LOG_DIR, "phase2_monitor.csv")
        env = CustomEnv(size=1000, max_steps=MAX_STEPS_PER_EPISODE)
        env.target_move_frequency = 10  # Target se mueve cada 10 pasos
        env = Monitor(env, filename=monitor_filename)
        return env

    env_phase2 = DummyVecEnv([make_env_moving])
    model.set_env(env_phase2)

    metrics_callback_phase2 = EpisodeMetricsCallback(verbose=1)
    model.learn(
        total_timesteps=TIMESTEPS_PHASE2, 
        callback=metrics_callback_phase2, 
        progress_bar=True, 
        reset_num_timesteps=False
    )

    # Guardar modelo final
    final_model_path = os.path.join(MODELS_DIR, "sac_cylinder_final")
    model.save(final_model_path)
    print(f"\n Modelo Final guardado en: {final_model_path}")

    # Guardar métricas de fase 2
    metrics_df_phase2 = metrics_callback_phase2.get_metrics_df()
    metrics_df_phase2['phase'] = 2
    metrics_path_phase2 = os.path.join(LOG_DIR, "phase2_metrics.csv")
    metrics_df_phase2.to_csv(metrics_path_phase2, index=False)
    print(f"Métricas Fase 2 guardadas en: {metrics_path_phase2}")
    
    plot_episode_metrics(metrics_df_phase2, save_suffix="_phase2", title_prefix="Fase 2 (Target Móvil)")

    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Episodios Fase 1: {len(metrics_df_phase1)}")
    print(f"Episodios Fase 2: {len(metrics_df_phase2)}")
    print(f"Total de episodios: {len(metrics_df_phase1) + len(metrics_df_phase2)}")
    print(f"{'='*70}\n")

    return model, metrics_df_phase1, metrics_df_phase2

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    model, metrics_phase1, metrics_phase2 = train_model_curriculum()
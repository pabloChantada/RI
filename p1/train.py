
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import CustomEnv

# Configuración de entrenamiento
MAX_STEPS_PER_EPISODE = 10   # Pasos por episodio
NUM_EPISODES = 15             # Número de episodios
# En el model.learn() el total_timesteps es el TOTAL de los steps de entrenamiento
# No son los steps por episodio
TOTAL_TIMESTEPS = MAX_STEPS_PER_EPISODE * NUM_EPISODES  # Total timesteps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FINAL_PATH = os.path.join(MODELS_DIR, "sac_cylinder")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class EpisodeMetricsCallback(BaseCallback):
    """
    Callback personalizado para registrar métricas por episodio
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_distances = []
        self.current_episode_reward = 0
        self.current_episode_distances = []
        
    # Esta funcion se llamada despues de cada env.step(). Esto nos permite
    # obtener las metricas especificas de cada episodio
    def _on_step(self):
        # Acumular recompensa del episodio paso actual
        # Con locals podemos obtener los valores del env
        self.current_episode_reward += self.locals['rewards'][0] 
        
        # Obtener información del entorno
        info = self.locals['infos'][0]
        if 'distance' in info:
            self.current_episode_distances.append(info['distance'])
        
        # Detectar fin de episodio
        done = self.locals['dones'][0]
        if done:
            # Guardar métricas del episodio
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
            
            # Reset para próximo episodio
            self.current_episode_reward = 0
            self.current_episode_distances = []
        
        return True
    
    def get_metrics_df(self):
        """Retorna DataFrame con métricas de todos los episodios"""
        data = {
            'episode': list(range(1, len(self.episode_rewards) + 1)),
            'total_reward': self.episode_rewards,
        }
        
        if self.episode_distances:
            data['avg_distance'] = [d['avg_distance'] for d in self.episode_distances]
            data['final_distance'] = [d['final_distance'] for d in self.episode_distances]
            data['initial_distance'] = [d['initial_distance'] for d in self.episode_distances]
        
        return pd.DataFrame(data)


def make_env():
    """Crea el entorno con configuración correcta"""
    monitor_filename = os.path.join(LOG_DIR, "train_monitor.csv")
    # Los MAX_STEPS_PER_EPISODE tienen que ser los mismo en todos los Env
    env = CustomEnv(size=1000, max_steps=MAX_STEPS_PER_EPISODE)
    env = Monitor(env, filename=monitor_filename)
    return env


def plot_episode_metrics(metrics_df):
    """Genera gráficas de métricas por episodio"""
    sns.set_style("whitegrid")
    
    # Usar GridSpec para layout flexible: 2 en la primera fila, 1 que ocupa toda la segunda fila
    fig = plt.figure(figsize=(14, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    fig.suptitle('Métricas de Entrenamiento por Episodio', fontsize=16, fontweight='bold')
    
    # 1. Recompensa total por episodio (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics_df['episode'], metrics_df['total_reward'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Episodio', fontweight='bold')
    ax1.set_ylabel('Recompensa Total', fontweight='bold')
    ax1.set_title('Recompensa Total por Episodio')
    ax1.grid(True, alpha=0.3)
    
    # 2. Mejora en distancia por episodio (top-right)
    if 'initial_distance' in metrics_df.columns and 'final_distance' in metrics_df.columns:
        ax2 = fig.add_subplot(gs[0, 1])
        improvement = metrics_df['initial_distance'] - metrics_df['final_distance']
        colors = ['green' if x > 0 else 'red' for x in improvement]
        ax2.bar(metrics_df['episode'], improvement, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Episodio', fontweight='bold')
        ax2.set_ylabel('Mejora en Distancia', fontweight='bold')
        ax2.set_title('Mejora en Distancia (Inicial - Final)')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, 'No hay datos de mejora', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Mejora en Distancia')
    
    # 3. Distancia al objetivo (bottom, spanning both columns)
    if 'avg_distance' in metrics_df.columns:
        ax3 = fig.add_subplot(gs[1, :])
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
        ax3 = fig.add_subplot(gs[1, :])
        ax3.text(0.5, 0.5, 'No hay datos de distancia', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Evolución de la Distancia')
    
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, "episode_metrics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nGráfica de métricas guardada en: {plot_path}")
    plt.close()

def train_model():
    """Entrena el modelo con configuración corregida"""
    print(f"\n{'='*60}")
    print(f"CONFIGURACIÓN DE ENTRENAMIENTO")
    print(f"{'='*60}")
    print(f"Episodios: {NUM_EPISODES}")
    print(f"Pasos por episodio: {MAX_STEPS_PER_EPISODE}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"{'='*60}\n")
    
    # Crear entorno
    env = DummyVecEnv([make_env])
    
    # Crear modelo SAC
    model = SAC(
        policy="MlpPolicy",
        env=env,
        gamma=0.99,
        learning_rate=3e-4,
        batch_size=64,  # Reducido para datasets pequeños
        ent_coef="auto",
        tau=0.005,
        target_update_interval=1,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
    )
    
    # Callback para métricas
    metrics_callback = EpisodeMetricsCallback(verbose=1)
    # Callback para obtener el mejor modelo 
    eval_callback = EvalCallback(
        env,
        best_model_save_path=CKPT_DIR,  
        log_path=LOG_DIR,
        eval_freq=max(50, TOTAL_TIMESTEPS // 3),
        n_eval_episodes=3,
        deterministic=False,
    )

    # Entrenar
    print("Iniciando entrenamiento\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[metrics_callback, eval_callback],
        progress_bar=True
    )
    
    # Guardar modelo
    model.save(FINAL_PATH)
    print(f"\nModelo guardado en: {FINAL_PATH}.zip")
    
    # Obtener y guardar métricas
    metrics_df = metrics_callback.get_metrics_df()
    metrics_path = os.path.join(LOG_DIR, "episode_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Métricas guardadas en: {metrics_path}")
    
    # Mostrar resumen
    print(f"\n{'='*60}")
    print(f"RESUMEN DEL ENTRENAMIENTO")
    print(f"{'='*60}")
    print(metrics_df.to_string(index=False))
    print(f"{'='*60}\n")
    
    # Generar gráficas
    plot_episode_metrics(metrics_df)
    
    return model, metrics_df


if __name__ == "__main__":
    model, metrics = train_model()


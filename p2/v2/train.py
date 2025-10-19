
"""
Entrenamiento NEAT para el entorno Robobo Red Cylinder Search
Con métricas detalladas y visualizaciones
"""
import os
import neat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from env import CustomEnv
from visualize import Visualizer
import pickle
from datetime import datetime

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

MAX_STEPS_PER_EPISODE = 30
MAX_EPISODES_PER_GENOME = 3  # Evaluar cada genoma múltiples veces

# Configuración de generaciones
N_GENERATIONS = 25 
CHECKPOINT_INTERVAL = 5

print(f"\n{'='*70}")
print(f"NEAT TRAINING CONFIGURATION")
print(f"{'='*70}")
print(f"Steps per episode: {MAX_STEPS_PER_EPISODE}")
print(f"Episodes per genome: {MAX_EPISODES_PER_GENOME}")
print(f"Generations: {N_GENERATIONS}")
print(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
print(f"{'='*70}\n")

# ============================================================================
# DIRECTORIES
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_neat")
LOG_DIR = os.path.join(BASE_DIR, "logs_neat")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_neat")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# METRICS TRACKING
# ============================================================================

class NEATMetricsTracker:
    """
    Tracker para métricas de entrenamiento NEAT.
    
    Tracks:
        - Fitness por generación (best, avg, worst)
        - Métricas de episodios (reward, distance)
        - Información de especies
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.generation_metrics = []
        self.episode_metrics = []
        self.current_generation = 0
        
    def add_generation_metrics(self, generation, best_fitness, avg_fitness, 
                               worst_fitness, num_species):
        """
        Add metrics for a generation.
        
        Args:
            generation: Generation number
            best_fitness: Best fitness in generation
            avg_fitness: Average fitness in generation
            worst_fitness: Worst fitness in generation
            num_species: Number of species
        """
        self.generation_metrics.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'worst_fitness': worst_fitness,
            'num_species': num_species
        })
        
    def add_episode_metrics(self, generation, genome_id, episode, 
                           reward, initial_distance, final_distance, 
                           avg_distance, steps, goal_reached,
                           initial_pos=None, final_pos=None, target_pos=None):
        """
        Add metrics for a single episode.
        
        Args:
            generation: Generation number
            genome_id: Genome identifier
            episode: Episode number for this genome
            reward: Total episode reward
            initial_distance: Initial distance to target
            final_distance: Final distance to target
            avg_distance: Average distance during episode
            steps: Number of steps taken
            goal_reached: Whether goal was reached
            initial_pos: Initial (x, z) position
            final_pos: Final (x, z) position
            target_pos: Target (x, z) position
        """
        metrics = {
            'generation': generation,
            'genome_id': genome_id,
            'episode': episode,
            'reward': reward,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'avg_distance': avg_distance,
            'distance_improvement': initial_distance - final_distance,
            'steps': steps,
            'goal_reached': goal_reached
        }
        
        # Add position data if provided
        if initial_pos is not None:
            metrics['initial_x'] = initial_pos[0]
            metrics['initial_z'] = initial_pos[1]
        if final_pos is not None:
            metrics['final_x'] = final_pos[0]
            metrics['final_z'] = final_pos[1]
        if target_pos is not None:
            metrics['target_x'] = target_pos[0]
            metrics['target_z'] = target_pos[1]
        
        self.episode_metrics.append(metrics)
    
    def get_generation_df(self):
        """Get generation metrics as DataFrame."""
        return pd.DataFrame(self.generation_metrics)
    
    def get_episode_df(self):
        """Get episode metrics as DataFrame."""
        return pd.DataFrame(self.episode_metrics)
    
    def save_metrics(self, prefix=""):
        """
        Save all metrics to CSV files.
        
        Args:
            prefix: Prefix for filenames
        """
        gen_df = self.get_generation_df()
        ep_df = self.get_episode_df()
        
        if not gen_df.empty:
            gen_path = os.path.join(LOG_DIR, f"{prefix}generation_metrics.csv")
            gen_df.to_csv(gen_path, index=False)
            print(f"Generation metrics saved to: {gen_path}")
        
        if not ep_df.empty:
            ep_path = os.path.join(LOG_DIR, f"{prefix}episode_metrics.csv")
            ep_df.to_csv(ep_path, index=False)
            print(f"Episode metrics saved to: {ep_path}")

# Global metrics tracker
metrics_tracker = NEATMetricsTracker()

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def eval_genome(genome, config, env, generation):
    """
    Evalúa un único genoma en el entorno.
    
    Args:
        genome: Genoma NEAT a evaluar
        config: Configuración NEAT
        env: Entorno de Robobo
        generation: Número de generación actual
    
    Returns:
        float: Fitness acumulado del genoma
    """
    # Crear red neuronal desde el genoma
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_fitness = 0.0
    
    # Evaluar el genoma en múltiples episodios
    for episode in range(MAX_EPISODES_PER_GENOME):
        observation, info = env.reset()
        episode_reward = 0.0
        episode_distances = [info['distance']]
        initial_distance = info['distance']
        initial_pos = info['agent_position']
        target_pos = info['target_position']
        done = False
        
        step = 0
        while not done and step < MAX_STEPS_PER_EPISODE:
            # Activar la red neuronal con la observación actual
            action = net.activate(observation)
            
            # Asegurar que la acción esté en el rango correcto [-1, 1]
            action = np.clip(action, -1.0, 1.0)
            
            # Ejecutar acción en el entorno
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_distances.append(info['distance'])
            done = terminated or truncated
            step += 1
        
        # Métricas del episodio
        final_distance = info['distance']
        final_pos = info['agent_position']
        avg_distance = np.mean(episode_distances)
        goal_reached = terminated
        
        total_fitness += episode_reward
        
        # Guardar métricas del episodio (incluyendo posiciones)
        metrics_tracker.add_episode_metrics(
            generation=generation,
            genome_id=genome.key,
            episode=episode + 1,
            reward=episode_reward,
            initial_distance=initial_distance,
            final_distance=final_distance,
            avg_distance=avg_distance,
            steps=step,
            goal_reached=goal_reached,
            initial_pos=initial_pos,
            final_pos=final_pos,
            target_pos=target_pos
        )
        
        print(f"  Episode {episode+1}/{MAX_EPISODES_PER_GENOME}: "
              f"Reward={episode_reward:.2f}, Steps={step}, "
              f"Dist: {initial_distance:.1f}→{final_distance:.1f}, "
              f"Goal={'✓' if goal_reached else '✗'}")
    
    # Fitness promedio de todos los episodios
    avg_fitness = total_fitness / MAX_EPISODES_PER_GENOME
    
    return avg_fitness

def eval_genomes(genomes, config):
    """
    Evalúa todos los genomas de una generación.
    
    Args:
        genomes: Lista de tuplas (genome_id, genome)
        config: Configuración NEAT
    """
    # Obtener generación actual desde el tracker
    generation = metrics_tracker.current_generation
    
    print(f"\n{'='*70}")
    print(f"GENERATION {generation}")
    print(f"{'='*70}")
    
    # Crear una única instancia del entorno para todos los genomas
    env = CustomEnv(max_steps=MAX_STEPS_PER_EPISODE)
    
    try:
        fitness_values = []
        
        for idx, (genome_id, genome) in enumerate(genomes, 1):
            print(f"\n--- Evaluating Genome {genome_id} ({idx}/{len(genomes)}) ---")
            genome.fitness = eval_genome(genome, config, env, generation)
            fitness_values.append(genome.fitness)
            print(f"Genome {genome_id} Final Fitness: {genome.fitness:.2f}")
        
        # Guardar métricas de la generación
        if fitness_values:
            metrics_tracker.add_generation_metrics(
                generation=generation,
                best_fitness=max(fitness_values),
                avg_fitness=np.mean(fitness_values),
                worst_fitness=min(fitness_values),
                num_species=len(set(g.species_id for _, g in genomes if hasattr(g, 'species_id')))
            )
        
        print(f"\n{'='*70}")
        print(f"Generation {generation} Summary:")
        print(f"  Best Fitness:  {max(fitness_values):.2f}")
        print(f"  Avg Fitness:   {np.mean(fitness_values):.2f}")
        print(f"  Worst Fitness: {min(fitness_values):.2f}")
        print(f"{'='*70}\n")
        
    finally:
        # Cerrar el entorno al finalizar
        env.close()
    
    # Incrementar contador de generación
    metrics_tracker.current_generation += 1

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_metrics(gen_df, ep_df):
    """
    Plot simplified training metrics in a single figure.
    
    Creates 4 essential plots:
    1. Fitness evolution over generations
    2. Distance evolution over generations  
    3. Success rate over generations
    4. Spatial distribution of final positions (ant-like visualization)
    
    Args:
        gen_df: DataFrame with generation metrics
        ep_df: DataFrame with episode metrics
    """
    if gen_df.empty or ep_df.empty:
        print("No metrics to plot")
        return
    
    # Aggregate episode data by generation
    gen_ep_metrics = ep_df.groupby('generation').agg({
        'final_distance': 'mean',
        'goal_reached': 'sum'
    }).reset_index()
    
    total_episodes_per_gen = ep_df.groupby('generation').size().values
    gen_ep_metrics['success_rate'] = (gen_ep_metrics['goal_reached'] / total_episodes_per_gen * 100)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NEAT Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Fitness Evolution
    ax1 = axes[0, 0]
    ax1.plot(gen_df['generation'], gen_df['best_fitness'], 
             marker='o', label='Best', linewidth=2.5, markersize=7, color='green')
    ax1.plot(gen_df['generation'], gen_df['avg_fitness'], 
             marker='s', label='Average', linewidth=2.5, markersize=7, color='blue')
    ax1.set_xlabel('Generation', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Fitness', fontweight='bold', fontsize=12)
    ax1.set_title('Fitness Evolution', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance Evolution
    ax2 = axes[0, 1]
    ax2.plot(gen_ep_metrics['generation'], gen_ep_metrics['final_distance'], 
             marker='o', linewidth=2.5, markersize=7, color='red')
    ax2.set_xlabel('Generation', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Final Distance', fontweight='bold', fontsize=12)
    ax2.set_title('Average Final Distance to Target', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success Rate
    ax3 = axes[1, 0]
    ax3.plot(gen_ep_metrics['generation'], gen_ep_metrics['success_rate'], 
             marker='o', linewidth=2.5, markersize=7, color='purple')
    ax3.set_xlabel('Generation', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Goal Achievement Rate', fontweight='bold', fontsize=13)
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Spatial Distribution (Ant Colony style)
    ax4 = axes[1, 1]
    
    # Check if position data exists
    if 'final_x' in ep_df.columns and 'final_z' in ep_df.columns:
        # Draw arena boundaries
        arena_size = 1000
        ax4.plot([-arena_size, arena_size, arena_size, -arena_size, -arena_size],
                [-arena_size, -arena_size, arena_size, arena_size, -arena_size],
                'k-', linewidth=2, label='Arena')
        
        # Get unique generations for color mapping
        generations = sorted(ep_df['generation'].unique())
        n_gens = len(generations)
        
        # Create colormap from early (red) to late (green) generations
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_gens))
        
        # Plot final positions colored by generation
        for idx, gen in enumerate(generations):
            gen_data = ep_df[ep_df['generation'] == gen]
            
            # Separate successful and failed attempts
            successes = gen_data[gen_data['goal_reached'] == True]
            failures = gen_data[gen_data['goal_reached'] == False]
            
            # Plot failures (smaller, more transparent)
            if not failures.empty:
                ax4.scatter(failures['final_x'], failures['final_z'],
                          c=[colors[idx]], s=30, alpha=0.3, marker='x',
                          edgecolors='none')
            
            # Plot successes (larger, more opaque)
            if not successes.empty:
                ax4.scatter(successes['final_x'], successes['final_z'],
                          c=[colors[idx]], s=60, alpha=0.6, marker='o',
                          edgecolors='black', linewidths=0.5)
        
        # Plot target position (if available)
        if 'target_x' in ep_df.columns and 'target_z' in ep_df.columns:
            target_x = ep_df['target_x'].iloc[0]
            target_z = ep_df['target_z'].iloc[0]
            ax4.scatter(target_x, target_z, c='red', s=300, marker='*', 
                       edgecolors='darkred', linewidths=2, label='Target', zorder=10)
            
            # Draw goal radius
            goal_circle = plt.Circle((target_x, target_z), 150, 
                                    color='red', fill=False, 
                                    linestyle='--', linewidth=2, alpha=0.5)
            ax4.add_patch(goal_circle)
        
        ax4.set_xlabel('X Position', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Z Position', fontweight='bold', fontsize=12)
        ax4.set_title('Final Positions Distribution\n(Early=Red → Late=Green, ✓=Success, ✗=Fail)', 
                     fontweight='bold', fontsize=13)
        ax4.set_xlim([-arena_size-100, arena_size+100])
        ax4.set_ylim([-arena_size-100, arena_size+100])
        ax4.set_aspect('equal')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for generations
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                                   norm=plt.Normalize(vmin=min(generations), 
                                                     vmax=max(generations)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax4, label='Generation')
        cbar.set_label('Generation', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No position data available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Final Positions Distribution', fontweight='bold', fontsize=13)
    
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, "training_metrics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training metrics plot saved at: {plot_path}")
    plt.close()

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def run(config_file, n_generations=50, checkpoint_interval=5):
    """
    Ejecuta el entrenamiento evolutivo NEAT.
    
    Args:
        config_file: Ruta al archivo de configuración NEAT
        n_generations: Número de generaciones a entrenar
        checkpoint_interval: Frecuencia de guardado de checkpoints
    """
    # Cargar configuración
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    
    # Crear población
    p = neat.Population(config)
    
    # Añadir reporteros
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Configurar checkpoints
    p.add_reporter(neat.Checkpointer(
        checkpoint_interval,
        filename_prefix=os.path.join(CKPT_DIR, 'neat-checkpoint-')
    ))
    
    # Ejecutar evolución
    print(f"\n{'='*70}")
    print(f"STARTING NEAT EVOLUTION")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    winner = p.run(eval_genomes, n_generations)
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Guardar el mejor genoma
    winner_file = os.path.join(MODELS_DIR, 'winner_genome.pkl')
    with open(winner_file, 'wb') as f:
        pickle.dump(winner, f)
    print(f"\n✓ Winner genome saved to: {winner_file}")
    
    # Guardar métricas
    metrics_tracker.save_metrics()
    
    # Mostrar información del ganador
    print('\n' + '='*70)
    print('BEST GENOME:')
    print('='*70)
    print(winner)
    print(f"\nTraining time: {training_time/60:.2f} minutes")
    
    # Evaluar el ganador
    print('\n' + '='*70)
    print('TESTING WINNER:')
    print('='*70)
    test_winner(winner, config)
    
    # Crear visualizaciones
    create_visualizations(config, winner, stats)
    
    return winner, stats

def test_winner(winner, config, n_episodes=5):
    """
    Prueba el genoma ganador en varios episodios.
    
    Args:
        winner: Genoma ganador
        config: Configuración NEAT
        n_episodes: Número de episodios de prueba
    """
    env = CustomEnv(max_steps=MAX_STEPS_PER_EPISODE)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    results = {
        'rewards': [],
        'distances': [],
        'successes': 0
    }
    
    try:
        for episode in range(n_episodes):
            observation, info = env.reset()
            episode_reward = 0.0
            done = False
            step = 0
            initial_distance = info['distance']
            
            print(f"\n--- Test Episode {episode+1}/{n_episodes} ---")
            
            while not done and step < MAX_STEPS_PER_EPISODE:
                action = net.activate(observation)
                action = np.clip(action, -1.0, 1.0)
                
                observation, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                done = terminated or truncated
                step += 1
            
            final_distance = info['distance']
            goal_reached = terminated
            
            results['rewards'].append(episode_reward)
            results['distances'].append(final_distance)
            if goal_reached:
                results['successes'] += 1
            
            print(f"Episode finished: Reward={episode_reward:.2f}, "
                  f"Steps={step}, "
                  f"Dist: {initial_distance:.1f}→{final_distance:.1f}, "
                  f"Goal={'✓' if goal_reached else '✗'}")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY ({n_episodes} episodes):")
        print(f"  Success Rate: {results['successes']}/{n_episodes} ({100*results['successes']/n_episodes:.1f}%)")
        print(f"  Avg Reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
        print(f"  Avg Final Distance: {np.mean(results['distances']):.1f} ± {np.std(results['distances']):.1f}")
        print(f"{'='*70}\n")
    
    finally:
        env.close()

def create_visualizations(config, winner, stats):
    """
    Crea visualizaciones del entrenamiento y la red ganadora.
    
    Args:
        config: Configuración NEAT
        winner: Genoma ganador
        stats: Estadísticas del entrenamiento
    """
    print("\nGenerating visualizations...")
    
    # Nombres de nodos para visualización
    node_names = {
        -1: 'Agent X',
        -2: 'Agent Z',
        -3: 'Target X',
        -4: 'Target Z',
        -5: 'Blob Visible',
        -6: 'Blob X',
        -7: 'Blob Y',
        -8: 'Blob Size',
        0: 'Left Wheel',
        1: 'Right Wheel'
    }
    
    # Red neuronal (sin poda)
    Visualizer.draw_net(
        config, winner, True,
        node_names=node_names,
        prune_unused=False,
        filename=os.path.join(LOG_DIR, "winner_net_full")
    )
    
    # Red neuronal (con poda)
    Visualizer.draw_net(
        config, winner, True,
        node_names=node_names,
        prune_unused=True,
        filename=os.path.join(LOG_DIR, "winner_net_pruned")
    )
    
    # Gráficas de estadísticas NEAT
    Visualizer.plot_stats(
        stats,
        ylog=False,
        view=False,
        filename=os.path.join(LOG_DIR, "neat_fitness_stats.svg")
    )
    
    Visualizer.plot_species(
        stats,
        view=False,
        filename=os.path.join(LOG_DIR, "neat_species.svg")
    )
    
    # Gráficas de métricas personalizadas (simplificadas)
    gen_df = metrics_tracker.get_generation_df()
    ep_df = metrics_tracker.get_episode_df()
    
    plot_training_metrics(gen_df, ep_df)
    
    print(f"✓ All visualizations saved in: {LOG_DIR}/")

def resume_from_checkpoint(checkpoint_file, n_generations=50):
    """
    Resume el entrenamiento desde un checkpoint.
    
    Args:
        checkpoint_file: Ruta al archivo de checkpoint
        n_generations: Número adicional de generaciones
    """
    p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    
    # Añadir reporteros
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Continuar evolución
    winner = p.run(eval_genomes, n_generations)
    
    return winner, stats

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Obtener ruta del archivo de configuración
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    # Verificar que existe el archivo de configuración
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        print("Please create config.txt first")
        exit(1)
    
    # Ejecutar entrenamiento
    winner, stats = run(config_path, n_generations=N_GENERATIONS, 
                       checkpoint_interval=CHECKPOINT_INTERVAL)
    
    # Para resumir desde checkpoint, descomenta y ajusta:
    # winner, stats = resume_from_checkpoint(
    #     'checkpoints_neat/neat-checkpoint-25',
    #     n_generations=25
    # )



"""
Script para probar un genoma guardado en el entorno Robobo
"""
import os
import neat
import numpy as np
import pickle
from env import CustomEnv

def load_genome(genome_path):
    """Carga un genoma desde archivo."""
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    return genome

def test_genome(genome, config, n_episodes=20, max_steps=30, visualize=True):
    """
    Prueba un genoma en múltiples episodios.
    
    Args:
        genome: Genoma NEAT a probar
        config: Configuración NEAT
        n_episodes: Número de episodios de prueba
        max_steps: Pasos máximos por episodio
        visualize: Si mostrar información detallada
    """
    env = CustomEnv(max_steps=max_steps)
    # Cargamos la red con la configuracion establecida
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    successes = 0
    total_rewards = []
    total_distances = []
    
    try:
        for episode in range(n_episodes):
            observation, info = env.reset()
            episode_reward = 0.0
            done = False
            step = 0
            
            if visualize:
                print(f"\n{'='*60}")
                print(f"Episode {episode+1}/{n_episodes}")
                print(f"{'='*60}")
                print(f"Initial Distance: {info['distance']:.1f}")
            
            while not done and step < max_steps:
                # Activar red neuronal
                action = net.activate(observation)
                action = np.clip(action, -1.0, 1.0)
                
                # Ejecutar acción
                observation, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                done = terminated or truncated
                step += 1
            
            # Estadísticas del episodio
            final_distance = info['distance']
            goal_reached = terminated
            
            if goal_reached:
                successes += 1
            
            total_rewards.append(episode_reward)
            total_distances.append(final_distance)
            
            if visualize:
                print(f"\n{'='*60}")
                print(f"Episode {episode+1} Results:")
                print(f"  - Reward: {episode_reward:.2f}")
                print(f"  - Steps: {step}")
                print(f"  - Final Distance: {final_distance:.1f}")
                print(f"  - Goal Reached: {'YES' if goal_reached else 'NO'}")
                print(f"{'='*60}")
    
    finally:
        env.close()
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"SUMMARY ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Success Rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Final Distance: {np.mean(total_distances):.1f} ± {np.std(total_distances):.1f}")
    print(f"{'='*60}\n")
    
    return {
        'success_rate': successes / n_episodes,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_distance': np.mean(total_distances),
        'std_distance': np.std(total_distances)
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test a saved NEAT genome')
    parser.add_argument(
        '--genome',
        type=str,
        default='checkpoints/winner_genome.pkl',
        help='Path to saved genome file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.txt',
        help='Path to NEAT config file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of test episodes'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=30,
        help='Maximum steps per episode'
    )
    
    args = parser.parse_args()
    
    # Verificar archivos
    if not os.path.exists(args.genome):
        print(f"Error: Genome file not found at {args.genome}")
        exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    
    # Cargar configuración y genoma
    print("Loading configuration...")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config
    )
    
    print(f"Loading genome from {args.genome}...")
    genome = load_genome(args.genome)
    
    # Probar genoma
    results = test_genome(
        genome,
        config,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        visualize=True
    )

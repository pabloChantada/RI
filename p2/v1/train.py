import os
import neat
import pickle
from datetime import datetime
from env import CustomEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_neat")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_neat")
LOG_DIR = os.path.join(BASE_DIR, "logs_neat")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


POP_SIZE = 20
GENERATIONS = 30
EPISODES_PER_GENOME = 1
MAX_STEPS = 25


def evaluate_genomes(genomes, config):
    """
    Evaluates genomes sequentially. Each entry in genomes is (genome_id, genome_obj).
    For each genome we run EPISODES_PER_GENOME episodes and set genome.fitness
    to the average episode reward.
    """
    print(f"\n[INFO] Evaluating {len(genomes)} genomes...")

    env = CustomEnv(max_steps=MAX_STEPS, verbose=False)
    for idx, (genome_id, genome) in enumerate(genomes):
        env.reset()
        # Crear env por evaluación (evita compartir estados)
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        total_f = 0.0
        try:
            for ep in range(EPISODES_PER_GENOME):
                obs, info = env.reset()
                done = False
                steps = 0
                ep_reward = 0.0

                while not done and steps < MAX_STEPS:
                    out = net.activate(obs)
                    # Asegurar longitud correcta (2 salidas) y tipo float
                    action = [float(a) for a in out]
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                    steps += 1

                total_f += ep_reward

        # finally:
        #     env.close()
        except Exception as e:
            print(
                f"[ERROR] Exception during genome {genome_id} evaluation: {type(e).__name__}: {str(e)}"
            )

        # fitness: promedio de episodios. NEAT busca valores mayores si fitness_criterion = max
        genome.fitness = total_f / EPISODES_PER_GENOME

        # Log cada 10 genomas para no saturar la salida
        # if (idx + 1) % 10 == 0:
        print(
            f"[INFO] Evaluated {idx + 1}/{len(genomes)} genomes. Last fitness: {genome.fitness:.2f}"
        )


class SimpleCheckpointer(neat.reporting.BaseReporter):
    """
    Minimal checkpointer that avoids pickling the NEAT Config object (which may
    include non-picklable items like itertools.count). We only save the data
    necessary to resume: generation, population, and species. This mirrors the
    default Checkpointer but excludes the Config object to avoid the
    'cannot pickle itertools.count' error.
    """

    def __init__(self, generation_interval=1, filename_prefix="neat-checkpoint-"):
        self.generation_interval = generation_interval
        self.filename_prefix = filename_prefix
        self.current_generation = 0

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        if (self.current_generation + 1) % self.generation_interval != 0:
            return
        filename = f"{self.filename_prefix}{self.current_generation}"
        path = os.path.join(CKPT_DIR, filename)
        data = {
            "generation": self.current_generation,
            # population is a dict of genomes; it's safe to pickle genomes themselves
            # but avoid including the full config object.
            "population": population,
            # species_set may contain references to genomes; include it as-is
            "species_set": species_set,
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[CHECKPOINT] Saved simple checkpoint to {path}")
        except Exception as e:
            print(f"[CHECKPOINT ERROR] Failed to save checkpoint: {e}")


def run(config_path):
    print("=" * 70)
    print("[INFO] Starting NEAT Training")
    print("=" * 70)
    print(f"[CONFIG] Config file: {config_path}")
    print(f"[CONFIG] Models directory: {MODELS_DIR}")
    print(f"[CONFIG] Checkpoints directory: {CKPT_DIR}")
    print(f"[CONFIG] Logs directory: {LOG_DIR}")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    print(f"[CONFIG] Population size: {POP_SIZE}")
    print(f"[CONFIG] Generations: {GENERATIONS}")
    print(f"[CONFIG] Episodes per genome: {EPISODES_PER_GENOME}")
    print(f"[CONFIG] Max steps per episode: {MAX_STEPS}")
    print(f"[CONFIG] Number of inputs: {config.genome_config.num_inputs}")
    print(f"[CONFIG] Number of outputs: {config.genome_config.num_outputs}")
    print(f"[CONFIG] Total steps: {POP_SIZE * EPISODES_PER_GENOME * GENERATIONS}")
    print("=" * 70)

    p = neat.Population(config)
    # Stats generadas por NEAT
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Replace the default Checkpointer (which pickles the full config and may fail)
    # with our SimpleCheckpointer that does not attempt to pickle the Config object.
    p.add_reporter(
        SimpleCheckpointer(generation_interval=5, filename_prefix="neat-checkpoint-")
    )

    print("\n[INFO] Starting evolution...")
    start = datetime.now()
    print(f"[INFO] Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Aquí usamos la función secuencial evaluate_genomes
    winner = p.run(evaluate_genomes, n=GENERATIONS)

    end = datetime.now()
    duration = (end - start).total_seconds()

    print("\n" + "=" * 70)
    print("[INFO] Training completed!")
    print("=" * 70)
    print(f"[RESULTS] End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"[RESULTS] Total training time: {duration / 60.0:.2f} minutes ({duration:.2f} seconds)"
    )
    print(f"[RESULTS] Winner fitness: {winner.fitness:.2f}")
    print(f"[RESULTS] Winner genome ID: {winner.key}")

    # Guardar ganador
    winner_path = os.path.join(MODELS_DIR, "winner_genome.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)

    print(f"[RESULTS] Winner genome saved to: {winner_path}")

    # Guardar también el config usado
    config_save_path = os.path.join(MODELS_DIR, "config_used.txt")
    import shutil

    shutil.copy(config_path, config_save_path)
    print(f"[RESULTS] Config file copied to: {config_save_path}")

    # Guardar estadísticas
    stats_path = os.path.join(MODELS_DIR, "stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"[RESULTS] Statistics saved to: {stats_path}")

    # Resumen final
    print("\n[SUMMARY] Best genome statistics:")
    print(f"  - Nodes: {len(winner.nodes)}")
    print(f"  - Connections: {len(winner.connections)}")
    print(f"  - Fitness: {winner.fitness:.2f}")
    print("=" * 70 + "\n")

    return winner, stats


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    # Por defecto busca config.txt en el mismo directorio
    config_path = os.path.join(local_dir, "config.txt")

    if not os.path.exists(config_path):
        print(f"[ERROR] config.txt not found in {local_dir}")
        exit(1)

    print(f"[INFO] Config file found: {config_path}\n")

    try:
        winner, stats = run(config_path)
        print("[SUCCESS] Training pipeline completed successfully!")
    except Exception as e:
        print("\n[ERROR] Training failed with exception:")
        print(f"[ERROR] {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)

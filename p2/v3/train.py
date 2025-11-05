import os
import neat
import pickle
import shutil
from datetime import datetime
from env import CustomEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_neat")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_neat")
LOG_DIR = os.path.join(BASE_DIR, "logs_neat")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


POP_SIZE = 50
GENERATIONS = 30
EPISODES_PER_GENOME = 1
MAX_STEPS = 100


def evaluate_genomes(genomes, config):
    """
    Evalúa genomas secuencialmente. Para cada genoma, ejecuta EPISODES_PER_GENOME
    episodios y asigna a genome.fitness la recompensa media.
    """
    print(f"\n[INFO] Evaluating {len(genomes)} genomes...")

    env = CustomEnv(max_steps=MAX_STEPS, verbose=False)
    for idx, (genome_id, genome) in enumerate(genomes):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_f = 0.0

        try:
            for _ in range(EPISODES_PER_GENOME):
                obs, _ = env.reset()
                ep_reward = 0.0
                done = False
                steps = 0
                while not done and steps < MAX_STEPS:
                    action = net.activate(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                    steps += 1
                total_f += ep_reward
        except Exception as e:
            print(f"[ERROR] Exception during genome {genome_id} evaluation: {e}")
            total_f = 0.0  # Penalizar si hay error

        genome.fitness = total_f / EPISODES_PER_GENOME
        if (idx + 1) % 10 == 0 or (idx + 1) == len(genomes):
            print(
                f"  [Progress] Evaluated {idx + 1}/{len(genomes)} genomes. Last fitness: {genome.fitness:.2f}"
            )


class SimpleCheckpointer(neat.reporting.BaseReporter):
    """
    Un checkpointer minimalista que evita el error 'cannot pickle itertools.count'
    al no guardar el objeto de configuración completo.
    """

    def __init__(self, generation_interval=5, filename_prefix="neat-checkpoint-"):
        self.generation_interval = generation_interval
        self.filename_prefix = filename_prefix
        self.current_generation = 0

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        if (self.current_generation + 1) % self.generation_interval != 0:
            return
        filename = os.path.join(
            CKPT_DIR, f"{self.filename_prefix}{self.current_generation}"
        )
        print(
            f"\n[CHECKPOINT] Saving checkpoint for generation {self.current_generation} to {filename}..."
        )
        with open(filename, "wb") as f:
            data = (self.current_generation, config, population, species_set)
            pickle.dump(data, f)


def run(config_path):
    print("=" * 70)
    print("[INFO] Starting NEAT Training")
    print("=" * 70)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Sobrescribir constantes globales si están en el config
    # global POP_SIZE, GENERATIONS, EPISODES_PER_GENOME, MAX_STEPS
    # POP_SIZE = getattr(config, "pop_size", POP_SIZE)
    # GENERATIONS = getattr(config, "num_generations", GENERATIONS)
    # Aquí puedes añadir más si las pones en el config

    print(f"[CONFIG] Population size: {POP_SIZE}")
    print(f"[CONFIG] Generations: {GENERATIONS}")
    print(f"[CONFIG] Episodes per genome: {EPISODES_PER_GENOME}")
    print(f"[CONFIG] Max steps per episode: {MAX_STEPS}")
    print("=" * 70)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(SimpleCheckpointer(generation_interval=5))

    print("\n[INFO] Starting evolution...")
    start_time = datetime.now()

    winner = p.run(evaluate_genomes, n=GENERATIONS)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("[INFO] Training completed!")
    print(f"Total training time: {duration / 60.0:.2f} minutes")
    print(f"Best genome fitness: {winner.fitness:.2f}")

    # Guardar resultados
    winner_path = os.path.join(MODELS_DIR, "winner_genome.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"Winner genome saved to: {winner_path}")

    stats_path = os.path.join(MODELS_DIR, "stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Statistics saved to: {stats_path}")

    config_save_path = os.path.join(MODELS_DIR, "config_used.txt")
    shutil.copy(config_path, config_save_path)
    print(f"Config file copied to: {config_save_path}")
    print("=" * 70)

    return winner, stats


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    if not os.path.exists(config_path):
        print(f"[ERROR] config.txt not found in {local_dir}")
        exit(1)

    try:
        run(config_path)
        print("\n[SUCCESS] Training pipeline completed successfully!")
    except Exception as e:
        print(f"\n[FATAL ERROR] Training failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

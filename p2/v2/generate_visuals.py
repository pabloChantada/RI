import os
import pickle
import neat
import argparse
from visualize import Visualizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_neat")
VIS_DIR = os.path.join(BASE_DIR, "visuals_neat")

os.makedirs(VIS_DIR, exist_ok=True)


def generate_visuals(genome_path, config_path, stats_path, results_path, view=False):
    """
    Genera todas las visualizaciones requeridas para la práctica.
    """
    print("=" * 70)
    print("[INFO] Generating visualizations...")
    print("=" * 70)

    # Cargar datos
    try:
        with open(genome_path, "rb") as f:
            winner = pickle.load(f)
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        print("[INFO] Loaded genome, stats, and config.")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # 1. Gráfica de Aprendizaje
    print("[INFO] Generating learning curve plot (avg_fitness.svg)...")
    fitness_plot_path = os.path.join(VIS_DIR, "avg_fitness.svg")
    Visualizer.plot_stats(stats, ylog=False, view=view, filename=fitness_plot_path)

    # 2. Gráfica de Especies
    print("[INFO] Generating speciation plot (speciation.svg)...")
    speciation_plot_path = os.path.join(VIS_DIR, "speciation.svg")
    Visualizer.plot_species(stats, view=view, filename=speciation_plot_path)

    # 3. Gráfica de Red Neuronal
    print("[INFO] Generating neural network graph (winner_net.svg)...")
    net_plot_path = os.path.join(
        VIS_DIR, "winner_net"
    )  # Sin extensión, graphviz la añade
    node_names = {
        -1: "Agent X",
        -2: "Agent Z",
        -3: "Target X",
        -4: "Target Z",
        -5: "Blob Visible",
        -6: "Blob X",
        -7: "Blob Y",
        -8: "Blob Size",
        0: "Left Wheel",
        1: "Right Wheel",
    }
    Visualizer.draw_net(
        config,
        winner,
        view=view,
        node_names=node_names,
        filename=net_plot_path,
        prune_unused=True,
    )

    # 4. Gráfica de Plano 2D Solución (Trayectoria)
    if os.path.exists(results_path):
        print("[INFO] Generating 2D trajectory plot (trajectory.svg)...")
        trajectory_plot_path = os.path.join(VIS_DIR, "trajectory.svg")
        Visualizer.plot_trajectory(
            results_path, view=view, filename=trajectory_plot_path
        )
    else:
        print(
            f"[WARNING] Validation results not found at {results_path}. Skipping trajectory plot."
        )

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Visualizations saved in: {VIS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all visualizations for the NEAT training results."
    )
    parser.add_argument(
        "--genome",
        default=os.path.join(MODELS_DIR, "winner_genome.pkl"),
        help="Path to the winner genome file.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(MODELS_DIR, "config_used.txt"),
        help="Path to the config file used for training.",
    )
    parser.add_argument(
        "--stats",
        default=os.path.join(MODELS_DIR, "stats.pkl"),
        help="Path to the statistics file.",
    )
    parser.add_argument(
        "--results",
        default="validation_results.pkl",
        help="Path to the validation results file (from validate.py).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="If set, display plots directly instead of just saving them.",
    )
    args = parser.parse_args()

    if not all(os.path.exists(p) for p in [args.genome, args.config, args.stats]):
        print("[ERROR] One or more required files (genome, config, stats) not found.")
        print(
            "[INFO] Please run train.py first to generate these files in the 'models_neat' directory."
        )
    else:
        generate_visuals(args.genome, args.config, args.stats, args.results, args.view)

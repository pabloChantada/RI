import argparse
import pickle
import neat
import numpy as np
from env import CustomEnv
import os
from stable_baselines3 import SAC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_neat")
VISAL_PATH = os.path.join(MODELS_DIR, "validation_results.pkl")
SAC_MODEL_PATH = os.path.join(BASE_DIR, "sac_cylinder_final.zip")
DEFAULT_GENOME_PATH = os.path.join(MODELS_DIR, "winner_genome.pkl")
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config.txt")


def load_genome(path):
    """Carga un genoma NEAT desde un fichero .pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ar_model(path):
    """Carga un modelo de AR (SAC) pre-entrenado."""
    return SAC.load(path)


def adapt_observation_for_ar(obs_12d):
    """
    Adapta observación de 12 dimensiones (con IR) a 8 dimensiones (sin IR)
    para compatibilidad con el modelo AR pre-entrenado.

    Entrada (12 dims):
        [agent_x, agent_z, target_x, target_z,
         blob_visible, blob_x, blob_y, blob_size,
         ir_front_c, ir_front_l, ir_front_r, ir_front_rr]

    Salida (8 dims):
        [agent_x, agent_z, target_x, target_z,
         blob_visible, blob_x, blob_y, blob_size]

    Args:
        obs_12d: Observación de 12 dimensiones del ambiente

    Returns:
        np.ndarray: Observación de 8 dimensiones para AR
    """
    # Simplemente tomamos las primeras 8 dimensiones (sin los IR)
    return obs_12d[:8].astype(np.float32)


def should_use_ar_with_safety(obs_12d, blob_visible, ir_threshold=0.1):
    """
    Decide si usar AR basándose en blob visible Y seguridad de colisión.

    Si hay un obstáculo muy cercano (IR > threshold), no usar AR incluso
    si el blob es visible, para permitir que AE maneje la evasión.

    Args:
        obs_12d: Observación de 12 dimensiones
        blob_visible: Si el blob es visible
        ir_threshold: Umbral de proximidad peligrosa

    Returns:
        bool: True si es seguro usar AR
    """
    if not blob_visible:
        return False

    # Extraer sensores IR (dimensiones 8-11)
    ir_sensors = obs_12d[8:12]
    max_ir = np.max(ir_sensors)

    # Si hay obstáculo cercano, no usar AR (dejar que AE evite)
    if max_ir > ir_threshold:
        return False

    return True


def validate(
    genome_path,
    config_path,
    episodes=10,
    max_steps=100,
    policy_type="ae",
    ar_model_path=None,
    use_safety_check=True,
):
    """
    Ejecuta la validación para un genoma NEAT (AE) o una política híbrida AE+AR.

    Args:
        policy_type (str): 'ae' para política evolutiva pura, 'ae_ar' para híbrida.
        ar_model_path (str): Ruta al modelo AR (SAC) pre-entrenado, requerido para 'ae_ar'.
        use_safety_check (bool): Si True, evita usar AR cuando hay obstáculos cercanos.
    """
    print("=" * 70)
    print("--- Iniciando Validación ---")
    print("=" * 70)
    print(f"[CONFIG] Tipo de política: {policy_type}")
    print(f"[CONFIG] Genoma AE: {genome_path}")
    print(f"[CONFIG] Config NEAT: {config_path}")
    print(f"[CONFIG] Episodios: {episodes}")
    print(f"[CONFIG] Max steps: {max_steps}")
    print(f"[CONFIG] Safety check: {use_safety_check}")

    ar_model = None
    if policy_type == "ae_ar":
        if not ar_model_path:
            raise ValueError(
                "La ruta del modelo AR es necesaria para la política 'ae_ar'."
            )
        print(f"[CONFIG] Modelo AR: {ar_model_path}")
        ar_model = load_ar_model(ar_model_path)
        print("[INFO] Modelo AR cargado correctamente")
        print("[INFO] Wrapper de adaptación 12D->8D activado para AR")

    # Cargar modelo AE (NEAT)
    print(f"[INFO] Cargando genoma desde {genome_path}...")
    genome = load_genome(genome_path)
    print(f"[INFO] Genoma cargado - ID: {genome.key}, Fitness: {genome.fitness}")

    print(f"[INFO] Cargando configuración NEAT desde {config_path}...")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    ae_net = neat.nn.FeedForwardNetwork.create(genome, config)
    print("[INFO] Red neuronal NEAT creada correctamente")
    print("=" * 70)

    results = []
    env = None

    # Contadores de uso de políticas
    ae_count = 0
    ar_count = 0
    ar_blocked_by_safety = 0

    try:
        env = CustomEnv(max_steps=max_steps, verbose=False)

        for ep in range(episodes):
            obs, info = env.reset()
            traj = []
            ep_reward = 0.0
            done = False
            step = 0

            print(f"\n{'=' * 70}")
            print(f"[EPISODE {ep + 1}/{episodes}] Iniciando episodio...")
            print(f"{'=' * 70}")

            while not done and step < max_steps:
                blob_visible = info.get("blob_visible", False)
                ir_sensors = info.get("ir_sensors", {})
                max_ir = max(ir_sensors.values()) if ir_sensors else 0.0

                # Selección de política con safety check
                if policy_type == "ae_ar" and ar_model is not None:
                    if use_safety_check:
                        # Usar AR solo si es seguro (blob visible Y sin obstáculos)
                        use_ar = should_use_ar_with_safety(obs, blob_visible)

                        if blob_visible and not use_ar:
                            ar_blocked_by_safety += 1
                    else:
                        # Sin safety check, usar AR siempre que blob sea visible
                        use_ar = blob_visible

                    if use_ar:
                        # Adaptar observación de 12D a 8D para AR
                        obs_8d = adapt_observation_for_ar(obs)
                        action, _ = ar_model.predict(obs_8d, deterministic=True)
                        policy_used = "AR"
                        ar_count += 1
                    else:
                        # Usar política AE
                        out = ae_net.activate(obs)
                        action = np.clip(out, -1.0, 1.0)
                        policy_used = "AE"
                        ae_count += 1
                else:
                    # Solo AE
                    out = ae_net.activate(obs)
                    action = np.clip(out, -1.0, 1.0)
                    policy_used = "AE"
                    ae_count += 1

                obs, reward, terminated, truncated, info = env.step(action)

                # Logging
                pos = info.get("agent_position", (None, None))
                dist = info.get("distance", None)
                near_obstacle = info.get("near_obstacle", False)

                if step % 10 == 0:
                    obstacle_warning = " ⚠️" if near_obstacle else ""
                    print(
                        f"  [STEP {step:3d}] Política={policy_used:2s} | "
                        f"BlobVisible={str(blob_visible):5s} | "
                        f"Distancia={dist:6.1f} | "
                        f"MaxIR={max_ir:4.2f}{obstacle_warning} | "
                        f"Reward={reward:6.2f}"
                    )

                traj.append(
                    {
                        "pos": pos,
                        "distance": dist,
                        "blob": info.get("blob_visible", False),
                        "policy": policy_used,
                        "near_obstacle": near_obstacle,
                        "max_ir": max_ir,
                    }
                )

                ep_reward += reward
                done = terminated or truncated
                step += 1

            results.append(
                {
                    "episode": ep + 1,
                    "reward": ep_reward,
                    "steps": step,
                    "trajectory": traj,
                    "reached": terminated,
                }
            )

            print(f"\n[EPISODE {ep + 1}] Finalizado:")
            print(f"  - Objetivo alcanzado: {terminated}")
            print(f"  - Pasos totales: {step}")
            print(f"  - Recompensa total: {ep_reward:.2f}")
            print("=" * 70)

    finally:
        if env is not None:
            env.close()

    # Guardar resultados
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(VISAL_PATH, "wb") as f:
        pickle.dump(results, f)

    print("\n" + "=" * 70)
    print("--- Validación Finalizada ---")
    print("=" * 70)
    print(f"[RESULTS] Resultados guardados en: {VISAL_PATH}")

    # Resumen de resultados
    total_reached = sum(1 for r in results if r["reached"])
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    print(f"[SUMMARY] Episodios completados: {episodes}")
    print(
        f"[SUMMARY] Objetivos alcanzados: {total_reached}/{episodes} "
        f"({100 * total_reached / episodes:.1f}%)"
    )
    print(f"[SUMMARY] Recompensa promedio: {avg_reward:.2f}")
    print(f"[SUMMARY] Pasos promedio: {avg_steps:.1f}")

    if policy_type == "ae_ar":
        total_actions = ae_count + ar_count
        print("\n[POLICY USAGE]")
        print(
            f"  - AE usada: {ae_count}/{total_actions} ({100 * ae_count / total_actions:.1f}%)"
        )
        print(
            f"  - AR usada: {ar_count}/{total_actions} ({100 * ar_count / total_actions:.1f}%)"
        )
        if use_safety_check:
            print(f"  - AR bloqueada por obstáculos: {ar_blocked_by_safety}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validar un genoma NEAT para Robobo.")
    parser.add_argument(
        "--genome",
        default=DEFAULT_GENOME_PATH,
        help="Ruta al fichero winner_genome.pkl.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Ruta al fichero de configuración de NEAT.",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Número de episodios a ejecutar."
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Máximo de pasos por episodio."
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="ae_ar",
        choices=["ae", "ae_ar"],
        help="Tipo de política: 'ae' (solo evolutiva) o 'ae-ar'/'ae_ar' (híbrida).",
    )
    parser.add_argument(
        "--ar_model_path",
        type=str,
        default=SAC_MODEL_PATH,
        help="Ruta al modelo AR (SAC) pre-entrenado. Requerido si --policy_type es 'ae_ar'.",
    )
    parser.add_argument(
        "--no-safety-check",
        action="store_true",
        help="Desactiva el safety check (usa AR incluso cerca de obstáculos).",
    )

    args = parser.parse_args()

    # Verificar que los archivos existen
    if not os.path.exists(args.genome):
        print(f"[ERROR] No se encontró el genoma en: {args.genome}")
        exit(1)

    if not os.path.exists(args.config):
        print(f"[ERROR] No se encontró el config en: {args.config}")
        exit(1)

    if args.policy_type == "ae_ar" and not os.path.exists(args.ar_model_path):
        print(f"[ERROR] No se encontró el modelo AR en: {args.ar_model_path}")
        exit(1)

    validate(
        genome_path=args.genome,
        config_path=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        policy_type=args.policy_type,
        ar_model_path=args.ar_model_path,
        use_safety_check=not args.no_safety_check,
    )

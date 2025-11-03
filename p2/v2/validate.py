import argparse
import pickle
import neat
import numpy as np
from env import CustomEnv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_neat")
VISUAL_PATH = os.path.join(MODELS_DIR, "validation_results.pkl")


def load_genome(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def validate(genome_path, config_path, episodes=5, max_steps=60, render=False):
    """
    Validation runner for an evolved NEAT genome (pure evolutionary policy).
    This version does NOT use reinforcement-learning or any AR fallback policy.
    It always uses the NEAT network to produce actions.

    Outputs:
      - validation_results.pkl with a list of episode dictionaries:
        { episode, reward, steps, trajectory, reached }
    """
    genome = load_genome(genome_path)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    results = []
    env = None
    try:
        for ep in range(episodes):
            env = CustomEnv(max_steps=max_steps, verbose=False)
            obs, info = env.reset()
            traj = []
            ep_reward = 0.0
            done = False
            step = 0
            while not done and step < max_steps:
                # Always use the evolved NEAT network (pure evolutionary policy)
                out = net.activate(obs)
                action = np.clip(out, -1.0, 1.0)
                obs, reward, terminated, truncated, info = env.step(action)
                pos = info.get("agent_position", (None, None))
                traj.append(
                    {
                        "pos": pos,
                        "distance": info.get("distance", None),
                        "blob": info.get("blob_visible", False),
                    }
                )
                ep_reward += reward
                done = terminated or truncated
                step += 1
            print(
                f"[INFO] Evaluated {ep + 1}/{episodes} episodes. Episode reward: {ep_reward:.2f} "
            )
            results.append(
                {
                    "episode": ep + 1,
                    "reward": ep_reward,
                    "steps": step,
                    "trajectory": traj,
                    "reached": terminated,
                }
            )
            env.close()
    finally:
        if env is not None:
            env.close()

    with open(VISUAL_PATH, "wb") as f:
        pickle.dump(results, f)
    print("Validation finished. Results saved to validation_results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", default="models_neat/winner_genome.pkl")
    parser.add_argument("--config", default="config.txt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=60)
    args = parser.parse_args()
    validate(
        args.genome,
        args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )

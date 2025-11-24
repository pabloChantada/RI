import numpy as np
from stable_baselines3 import SAC
from robobopy.utils.BlobColor import BlobColor

"""
ar_control_sim.py

Funciones para usar la política SAC entrenada en P1 dentro del simulador.

- get_blob_info_sim(robobo, sim): lee blob (readColorBlob) y posiciones desde sim.
- build_obs_8d_from_blob(blob_info, sim): construye la observación 8D igual que en p1/env.py.
- ar_policy_step(obs_8d): llama al modelo SAC cargado y devuelve la acción [a_left, a_right].
"""

# Ruta por defecto al modelo entrenado en P1 (ajusta si hace falta)
SAC_MODEL_PATH = "sac_cylinder_final.zip"
ar_model = SAC.load(SAC_MODEL_PATH, device="cpu")


def get_agent_position(sim):
    try:
        loc = sim.getRobotLocation(0)
        return loc["position"]["x"], loc["position"]["z"]
    except Exception:
        return 0.0, 0.0


def get_object_position(sim):
    try:
        obj_id = list(sim.getObjects())[0]
        loc = sim.getObjectLocation(obj_id)
        return loc["position"]["x"], loc["position"]["z"]
    except Exception:
        return 0.0, 0.0


def get_blob_info_sim(robobo, sim, goal_threshold=150.0):
    """
    Versión idéntica a la usada en p1/env.py:_get_blob_info(), pero sin dependencia del env.
    Devuelve dict con: visible (0/1), x,y (0-100), size (0-400 aprox), distance.
    """
    try:
        blob = robobo.readColorBlob(BlobColor.RED)
        agent_x, agent_z = get_agent_position(sim)
        target_x, target_z = get_object_position(sim)
        distance_to_target = np.sqrt(
            (agent_x - target_x) ** 2 + (agent_z - target_z) ** 2
        )

        if blob is None or blob.size <= 0:
            return {
                "visible": 0.0,
                "x": 50.0,
                "y": 50.0,
                "size": 0.0,
                "distance": distance_to_target,
            }

        if distance_to_target < goal_threshold:
            # consideramos objetivo alcanzado: blob centrado y grande
            return {
                "visible": 1.0,
                "x": 50.0,
                "y": 50.0,
                "size": 400.0,
                "distance": distance_to_target,
            }

        return {
            "visible": 1.0,
            "x": float(blob.posx),
            "y": float(blob.posy),
            "size": float(blob.size),
            "distance": distance_to_target,
        }

    except Exception:
        # En caso de error devolvemos "sin blob"
        return {"visible": 0.0, "x": 50.0, "y": 50.0, "size": 0.0, "distance": 9999.0}


def build_obs_8d_from_blob(blob_info, sim):
    """
    Construye la observación 8D igual que p1/env.py::_get_obs().
    """
    agent_x, agent_z = get_agent_position(sim)
    target_x, target_z = get_object_position(sim)

    agent_x_norm = np.clip(agent_x / 1000.0, -1.0, 1.0)
    agent_z_norm = np.clip(agent_z / 1000.0, -1.0, 1.0)
    target_x_norm = np.clip(target_x / 1000.0, -1.0, 1.0)
    target_z_norm = np.clip(target_z / 1000.0, -1.0, 1.0)

    blob_x_norm = np.clip((blob_info["x"] - 50.0) / 50.0, -1.0, 1.0)
    blob_y_norm = np.clip((blob_info["y"] - 50.0) / 50.0, -1.0, 1.0)
    blob_size_norm = np.clip(blob_info["size"] / 400.0, 0.0, 1.0)

    obs = np.array(
        [
            agent_x_norm,
            agent_z_norm,
            target_x_norm,
            target_z_norm,
            blob_info["visible"],
            blob_x_norm,
            blob_y_norm,
            blob_size_norm,
        ],
        dtype=np.float32,
    )
    return obs


def ar_policy_step(obs_8d: np.ndarray):
    """
    Ejecuta la política SAC (P1) sobre obs_8d y devuelve acción [a_left, a_right].
    """
    action, _ = ar_model.predict(obs_8d, deterministic=True)
    return action

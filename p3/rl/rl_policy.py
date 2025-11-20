
######################################################### POLÍTICA RL #####################################################################

from typing import Tuple

import numpy as np

import config


### DISCRETIZACIÓN ###

def _digitize(value: float, bins):
    """
    Devuelve el índice del bin en el que cae value.
    np.digitize pero con manejo simple.

    """
    return int(np.digitize([value], bins)[0])


def discretize_state(blob_size: float, blob_x: float, blob_y: float) -> Tuple[int, int, int]:
    """
    Convierte (blob_size, x, y) en índices discretos (isize, ix, iy).
    ADAPTA los bins de config a los de tu práctica 1 si eran distintos.

    """
    isize = _digitize(blob_size, config.SIZE_BINS)
    ix = _digitize(blob_x, config.X_BINS)
    iy = _digitize(blob_y, config.Y_BINS)
    return isize, ix, iy



### POLÍTICA HEURÍSTICA (BASELINE) ###

class HeuristicRLBaseline:
    """
    Política basada en reglas:
    - Si blob_x > 0: girar derecha
    - Si blob_x < 0: girar izquierda
    - Si blob_size es pequeño: avanzar
    - Si blob_size muy grande: parar

    """

    def predict(self, obs: np.ndarray) -> int:
        blob_size, blob_x, blob_y = obs

        if blob_size > config.TARGET_REACHED_BLOB_SIZE:
            return _action_index_from_name("STOP")

        # Corrige la orientación primero
        if blob_x > 0.2:
            return _action_index_from_name("RIGHT")
        elif blob_x < -0.2:
            return _action_index_from_name("LEFT")
        else:
            # Centrado: avanzar
            return _action_index_from_name("FORWARD")



### POLÍTICA REAL (PLANTILLA Q-TABLE) ###

class QTablePolicy:
    """

    Cargar la QTable (ns si estoy hay que tenerlo 100% pero bue)
   
    """

    def __init__(self, qtable_path: str):
        print(f"[RL] Cargando Q-table desde {qtable_path}")
        self.q_table = np.load(qtable_path)
        print(f"[RL] Q-table cargada con forma {self.q_table.shape}")

    def predict(self, obs: np.ndarray) -> int:
        blob_size, blob_x, blob_y = obs
        isize, ix, iy = discretize_state(blob_size, blob_x, blob_y)
        q_values = self.q_table[isize, ix, iy, :]
        return int(np.argmax(q_values))



### HELPERS ###


def _action_index_from_name(name: str) -> int:
    """
    
    Dado un nombre de acción ("FORWARD", etc.) devuelve el índice correspondiente.
    Si no existe, devuelve índice de STOP.

    """
    name = name.upper()
    for idx, n in config.RL_ACTION_MAPPING.items():
        if n == name:
            return idx
    # por defecto STOP
    for idx, n in config.RL_ACTION_MAPPING.items():
        if n == "STOP":
            return idx
    return 0


def load_policy():
    """

    Devuelve un objeto con método .predict(obs) compatible con Stable-Baselines.

    """
    if config.USE_HEURISTIC_POLICY:
        print("[RL] Usando política heurística de baseline (no RL real).")
        return HeuristicRLBaseline()
    else:
        # SUSTIUIR POR NUESTRA POLÍTICA
        return QTablePolicy(config.QTABLE_PATH)

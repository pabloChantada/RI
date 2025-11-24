import numpy as np
from ultralytics import YOLO
from stable_baselines3 import SAC

"""
ar_control_real.py

Funciones para:
 - detectar objeto con YOLO (sobre una imagen/stream real o robobo camera)
 - convertir bounding box -> features compatibles con la práctica 1 (obs 8D)
 - cargar y ejecutar la política SAC (ya entrenada) sobre esa observación

Rutas a modelos: ajusta SAC_MODEL_PATH y OBJECT_MODEL_PATH según tu árbol.
"""

# -----------------------------
# RUTAS / MODELOS
# -----------------------------
SAC_MODEL_PATH = "sac_cylinder_final.zip"  # ajustar si hace falta
OBJECT_MODEL_PATH = "yolo11n.pt"  # tu modelo de objetos (COCO o custom)

# Cargar modelos
ar_model = SAC.load(SAC_MODEL_PATH, device="cpu")
object_model = YOLO(OBJECT_MODEL_PATH)

# Ajustar según la clase que representa vuestro objeto en el modelo YOLO
TARGET_CLASS_ID = 0  # cambiar si no es la clase 0


# -----------------------------
# YOLO-OBJETO -> FEATURES
# -----------------------------
def extract_object_features(result, frame_width, frame_height):
    """
    A partir de la salida de YOLO-OBJETOS (un 'result' de ultralytics),
    devuelve:
        visible (bool),
        x_norm_01, y_norm_01: centro del bbox normalizado [0,1],
        size_norm_01: área normalizada [0,1],
        box_xyxy: (x1,y1,x2,y2) del bbox (opcional, para debug)
    """
    if result is None or len(result.boxes) == 0:
        return False, 0.0, 0.0, 0.0, None

    best_box = None
    best_conf = 0.0
    for box in result.boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if cls == TARGET_CLASS_ID and conf > best_conf:
            best_conf = conf
            best_box = box

    # si no hay caja de la clase objetivo
    if best_box is None:
        return False, 0.0, 0.0, 0.0, None

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    area = w * h

    x_norm = cx / frame_width
    y_norm = cy / frame_height
    size_norm = area / (frame_width * frame_height)

    return True, float(x_norm), float(y_norm), float(size_norm), (x1, y1, x2, y2)


def run_object_inference(frame, conf=0.5, imgsz=320):
    """
    Ejecuta YOLO-OBJETOS sobre un frame (numpy BGR) y devuelve:
        visible, x_n, y_n, size_n, result, box
    """
    results = object_model(frame, conf=conf, imgsz=imgsz, verbose=False)
    if len(results) == 0:
        return False, 0.0, 0.0, 0.0, None, None

    r0 = results[0]
    h, w, _ = frame.shape
    visible, x_n, y_n, size_n, box = extract_object_features(r0, w, h)
    return visible, x_n, y_n, size_n, r0, box


# -----------------------------
# BUILD OBS 8D (compatible P1)
# -----------------------------
def build_ar_observation_from_yolo(
    obj_visible,
    obj_x_norm_01,
    obj_y_norm_01,
    obj_size_norm_01,
    agent_pos=None,
    target_pos=None,
):
    """
    Construye la observación 8D usada en P1:
      [agent_x_norm, agent_z_norm, target_x_norm, target_z_norm,
       blob_visible, blob_x_norm, blob_y_norm, blob_size_norm]

    - obj_x_norm_01, obj_y_norm_01 en [0,1] (imagen)
    - obj_size_norm_01 en [0,1] (area relativa)
    - agent_pos/target_pos: opcional (x,z) reales para normalizar; si None -> zeros.
    """
    if agent_pos is None:
        agent_x_norm = 0.0
        agent_z_norm = 0.0
    else:
        ax, az = agent_pos
        agent_x_norm = np.clip(ax / 1000.0, -1.0, 1.0)
        agent_z_norm = np.clip(az / 1000.0, -1.0, 1.0)

    if target_pos is None:
        target_x_norm = 0.0
        target_z_norm = 0.0
    else:
        tx, tz = target_pos
        target_x_norm = np.clip(tx / 1000.0, -1.0, 1.0)
        target_z_norm = np.clip(tz / 1000.0, -1.0, 1.0)

    visible = 1.0 if obj_visible else 0.0

    # YOLO centro [0,1] -> blob_x_norm [-1,1] centrado (igual que p1)
    blob_x_norm = float(np.clip((obj_x_norm_01 - 0.5) * 2.0, -1.0, 1.0))
    blob_y_norm = float(np.clip((obj_y_norm_01 - 0.5) * 2.0, -1.0, 1.0))
    blob_size_norm = float(np.clip(obj_size_norm_01, 0.0, 1.0))

    obs_8d = np.array(
        [
            agent_x_norm,
            agent_z_norm,
            target_x_norm,
            target_z_norm,
            visible,
            blob_x_norm,
            blob_y_norm,
            blob_size_norm,
        ],
        dtype=np.float32,
    )
    return obs_8d


# -----------------------------
# AR policy step (obs 8D -> action)
# -----------------------------
def ar_policy_step_from_obs(obs_8d: np.ndarray):
    """
    Ejecuta la política SAC (P1) sobre obs_8d y devuelve acción [a_left, a_right]
    en [-1,1].
    """
    action, _ = ar_model.predict(obs_8d, deterministic=True)
    return action

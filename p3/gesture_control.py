import numpy as np
from ultralytics import YOLO

# -----------------------------
# MODELO DE POSE (YOLO)
# -----------------------------
POSE_MODEL_PATH = "yolo11n-pose.pt"
pose_model = YOLO(POSE_MODEL_PATH)


# -----------------------------
# DETECCIÓN DE GESTOS
# -----------------------------
def get_keypoint(kps, index):
    if kps is None or index >= len(kps):
        return None, None, 0.0
    return kps[index, 0], kps[index, 1], kps[index, 2]


def detect_gesture_from_pose(keypoints):
    """
    Gestos definidos:

        - brazos abajo              -> "STOP"
        - dos brazos arriba         -> "FORWARD"
        - solo brazo derecho arriba -> "RIGHT"
        - solo brazo izquierdo arriba -> "LEFT"

    Keypoints en formato COCO:
        5: left_shoulder,  6: right_shoulder
        9: left_wrist,    10: right_wrist
    """
    if keypoints is None:
        return None

    # hombros y muñecas
    l_sh_x, l_sh_y, c_lsh = get_keypoint(keypoints, 5)
    r_sh_x, r_sh_y, c_rsh = get_keypoint(keypoints, 6)
    l_wr_x, l_wr_y, c_lwr = get_keypoint(keypoints, 9)
    r_wr_x, r_wr_y, c_rwr = get_keypoint(keypoints, 10)

    # si hombros muy poco fiables, no inferimos gesto
    if c_lsh < 0.3 or c_rsh < 0.3:
        return None

    shoulder_dist = np.linalg.norm(
        np.array([l_sh_x, l_sh_y]) - np.array([r_sh_x, r_sh_y])
    )
    if shoulder_dist < 1e-5:
        return None

    left_up = False
    right_up = False

    if c_lwr > 0.3:
        # >0 si muñeca por encima del hombro
        rel_y_left = (l_sh_y - l_wr_y) / shoulder_dist
        left_up = rel_y_left > 0.4  # umbral ajustable

    if c_rwr > 0.3:
        rel_y_right = (r_sh_y - r_wr_y) / shoulder_dist
        right_up = rel_y_right > 0.4

    if left_up and right_up:
        return "FORWARD"
    elif right_up and not left_up:
        return "RIGHT"
    elif left_up and not right_up:
        return "LEFT"
    else:
        return "STOP"


def run_pose_inference(frame, conf=0.5, imgsz=320):
    """
    Ejecuta YOLO pose sobre un frame y devuelve:
        - keypoints (np.array [N_kps, 3]) o None
        - results (objeto ultralytics completo)
    """
    results = pose_model(frame, conf=conf, imgsz=imgsz, verbose=False)

    keypoints = None
    if len(results) > 0 and len(results[0].keypoints) > 0:
        kp = results[0].keypoints[0]
        keypoints = np.concatenate(
            [kp.xy[0].cpu().numpy(), kp.conf[0].cpu().numpy().reshape(-1, 1)],
            axis=1,
        )

    return keypoints, results

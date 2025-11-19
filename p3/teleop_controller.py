
######################################################### CONTROL DE TELEOPERADOR #####################################################################

# bua esto ns que es exactamente eh
# en plan me refiero a partes del script, creo que deberíamos cambiarlas (aunque le he preguntado a chati por el código original y ha dicho que no :I)

from typing import Optional

import numpy as np


"""
Indices de keypoints en YOLO11/YOLOv8 pose (COCO, 17 puntos) :contentReference[oaicite:1]{index=1}

0 Nose
1 Left Eye
2 Right Eye
3 Left Ear
4 Right Ear
5 Left Shoulder
6 Right Shoulder
7 Left Elbow
8 Right Elbow
9 Left Wrist
10 Right Wrist
11 Left Hip
12 Right Hip
13 Left Knee
14 Right Knee
15 Left Ankle
16 Right Ankle
"""


def infer_gesture_from_keypoints(
    keypoints_xy: Optional[np.ndarray],
    frame_shape, ) -> str:
    
    """
    Recibe los puntos clave de la primera persona (17,2) y decide un gesto:

    "FORWARD", "LEFT", "RIGHT", "BACKWARD", "STOP"

    Si no se ve persona o faltan puntos clave, devuelve "STOP".
    """

    if keypoints_xy is None:
        return "STOP"

    h, w = frame_shape[:2]

    try:
        l_shoulder = keypoints_xy[5]
        r_shoulder = keypoints_xy[6]
        l_wrist = keypoints_xy[9]
        r_wrist = keypoints_xy[10]
        l_hip = keypoints_xy[11]
        r_hip = keypoints_xy[12]
    except Exception:
        return "STOP"

    # Si alguna coordenada está a cero, mala detección
    if np.all(l_shoulder == 0) or np.all(r_shoulder == 0):
        return "STOP"

    # Márgenes relativos (no hipersensible)
    margin_y = 0.05 * h
    margin_x = 0.05 * w

    # Medias hombros/caderas (wtf esto lo copié directamente hay que cambiarlo)
    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
    hip_y = (l_hip[1] + r_hip[1]) / 2.0

    # --------- Gesto 1: avanzar = brazos claramente por encima de hombros ---------
    both_hands_up = (
        (l_wrist[1] < shoulder_y - margin_y)
        and (r_wrist[1] < shoulder_y - margin_y)
    )
    if both_hands_up:
        return "FORWARD"

    # --------- Gesto 2: retroceder = brazos claramente por debajo de caderas -----
    both_hands_down = (
        (l_wrist[1] > hip_y + margin_y)
        and (r_wrist[1] > hip_y + margin_y)
    )
    if both_hands_down:
        return "BACKWARD"

    # --------- Gesto 3: giro derecha = mano derecha extendida hacia la derecha ---
    right_hand_right = r_wrist[0] > (r_shoulder[0] + margin_x)
    if right_hand_right:
        return "RIGHT"

    # --------- Gesto 4: giro izquierda = mano izquierda extendida a la izquierda -
    left_hand_left = l_wrist[0] < (l_shoulder[0] - margin_x)
    if left_hand_left:
        return "LEFT"

    # Por defecto
    return "STOP"


def gesture_to_command(gesture: str) -> str:
    """
    Mapea gestos a comandos de movimiento.
    """
    g = gesture.upper()
    if g in ("FORWARD", "LEFT", "RIGHT", "BACKWARD", "STOP"):
        if g == "BACKWARD":
            return "BACKWARD"
        return g
    return "STOP"

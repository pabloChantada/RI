"""
Script principal para P2 en MUNDO REAL (o cámara real del Robobo).

Flujo:
 - Webcam (PC) con YOLO-Pose -> gestos -> mapeo a acción continua
 - Cámara del robot (o stream / otra webcam) con YOLO-OBJ -> obs 8D
 - Mientras NO esté centrado: modo GESTOS (aplican acciones continuas de gestos)
 - Al detectarse centrado: modo AR -> política SAC (ar_control_real) toma el control
"""

import cv2
import time
import numpy as np

# ajusta si vas a usar Robobo real o sim (si no lo tienes, puedes comentar las líneas)
try:
    from robobopy.Robobo import Robobo
    from robobosim.RoboboSim import RoboboSim  # opcional si usas sim

    HAVE_ROBO = True
except Exception:
    HAVE_ROBO = False

from gesture_control import run_pose_inference, detect_gesture_from_pose
import ar_control_real as ar_real

# -----------------------------
# CONFIG
# -----------------------------
# Cámaras:
WEBCAM_ID = 0  # cámara para detectar gestos (tu webcam)
ROBO_CAM_ID = 1  # cámara del robot o segundo dispositivo / rtsp stream
# Si vas a usar la cámara del Robobo real, reemplaza ROBO_CAM_ID por la API del robobo
# (ej: usar robobo.readCameraImage() en vez de cv2.VideoCapture)

CONF_POSE = 0.5
CONF_OBJ = 0.5
FRAME_SKIP_POSE = 0

# Parámetros de acción/unificación (MISMO formato P1)
SPEED = 20.0
WHEELS_TIME = 0.25

# Estado
mode = "GESTOS"
last_detected_gesture = "STOP"
last_keypoints = None
frame_count = 0

# ---------------------------------
# ROBOT (opcional)
# ---------------------------------
robobo = None
sim = None
if HAVE_ROBO:
    robobo = Robobo("localhost")
    robobo.connect()
    # si quieres la simulación (opcional)
    try:
        sim = RoboboSim("localhost")
        sim.connect()
    except Exception:
        sim = None
    # configurar cámara del robot sin bloquear
    try:
        robobo.movePanTo(0, 50, wait=False)
        robobo.moveTiltTo(90, 50, wait=False)
        time.sleep(0.2)
    except Exception:
        pass


# ===================================================
# Funciones de mapeo y aplicación de acciones (unificadas)
# ===================================================
def action_from_gesture(gesture: str) -> np.ndarray:
    if gesture == "FORWARD":
        return np.array([+0.5, +0.5], dtype=np.float32)
    elif gesture == "LEFT":
        return np.array([-0.4, +0.4], dtype=np.float32)
    elif gesture == "RIGHT":
        return np.array([+0.4, -0.4], dtype=np.float32)
    elif gesture == "STOP":
        return np.array([0.0, 0.0], dtype=np.float32)
    else:
        return np.array([0.0, 0.0], dtype=np.float32)


def apply_action(action: np.ndarray, source: str = "GESTOS"):
    action = np.asarray(action, dtype=np.float32).flatten()
    left_vel = float(np.clip(action[0], -1.0, 1.0))
    right_vel = float(np.clip(action[1], -1.0, 1.0))

    left_motor = int(np.clip(left_vel * SPEED, -100, 100))
    right_motor = int(np.clip(right_vel * SPEED, -100, 100))

    print(f"[{source}] action={action} -> L={left_motor} R={right_motor}")

    if HAVE_ROBO:
        # En P1 se llamaba: robobo.moveWheelsByTime(right_motor, left_motor, wheels_time)
        robobo.moveWheelsByTime(right_motor, left_motor, WHEELS_TIME)
    else:
        # si no hay robot: solo echo en consola (útil para pruebas en PC)
        pass


# ===================================================
# Lectura imagenes objeto (ROBÓBO camera) - fallback a cv2.VideoCapture
# ===================================================
def read_robo_frame():
    """
    Si tienes acceso directo a la imagen del Robobo, debes obtener la imagen desde
    la API (ej. robobo.readCameraImage()). Si no, como fallback usamos VideoCapture
    en ROBO_CAM_ID (útil para pruebas con una segunda webcam).
    """
    if HAVE_ROBO:
        try:
            # algunas APIS devuelven base64 o bytes; aquí asumimos que hay un método
            # que devuelve un numpy array BGR; ajusta según tu SDK.
            img = robobo.readCameraImage()
            return img
        except Exception:
            pass

    # fallback
    cap = getattr(read_robo_frame, "_cap", None)
    if cap is None:
        cap = cv2.VideoCapture(ROBO_CAM_ID)
        setattr(read_robo_frame, "_cap", cap)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


# ===================================================
# MAIN
# ===================================================
def main():
    global mode, last_detected_gesture, last_keypoints, frame_count

    # webcam para gestos (PC)
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("No se puede abrir la webcam de gestos")
        return

    # estado para forzar mínimo 5 acciones telecontroladas si se requiere
    tele_actions_count = 0
    MIN_TELE_ACTIONS = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # POSE (gestos) - solo cada N frames si quieres
            run_pose = (FRAME_SKIP_POSE == 0) or (
                frame_count % (FRAME_SKIP_POSE + 1) == 0
            )
            if run_pose:
                keypoints, _ = run_pose_inference(frame, conf=CONF_POSE, imgsz=320)
                last_keypoints = keypoints
                gesture = detect_gesture_from_pose(keypoints)
                if gesture is not None:
                    last_detected_gesture = gesture

            # LECTURA FRAME ROBOT / objeto
            robo_frame = read_robo_frame()
            obj_visible = False
            obj_x = obj_y = obj_size = 0.0
            if robo_frame is not None:
                obj_visible, obj_x, obj_y, obj_size, _, box = (
                    ar_real.run_object_inference(robo_frame, conf=CONF_OBJ, imgsz=320)
                )

            # CRITERIO cambio a AR: objeto visible y relativamente centrado.
            centered = obj_visible and (0.4 <= obj_x <= 0.6)

            if mode == "GESTOS":
                # aplicar la acción derivada del gesto (acción continua P1-like)
                action = action_from_gesture(last_detected_gesture)
                apply_action(action, source="GESTOS")
                tele_actions_count += 1

                if tele_actions_count >= MIN_TELE_ACTIONS and centered:
                    mode = "AR"
                    print("CAMBIO DE MODO -> AR (ejecuta política de refuerzo)")
                    # opcional: small wait to let robot settle
                    time.sleep(0.1)

            else:
                # modo AR: construir obs 8D desde YOLO-OBJ y (si tienes) posiciones reales
                # Si tienes posiciones reales (agent_pos / target_pos) pásalas; si no, usa None.
                obs_8d = ar_real.build_ar_observation_from_yolo(
                    obj_visible,
                    obj_x,
                    obj_y,
                    obj_size,
                    agent_pos=None,
                    target_pos=None,
                )
                action = ar_real.ar_policy_step_from_obs(obs_8d)
                apply_action(action, source="AR")
                # pausa para emular ritmo similar al env.step de P1
                time.sleep(0.08)

            # VISUAL DEBUG (muestra la webcam de gestos y overlay simple)
            vis = frame.copy()
            if last_keypoints is not None:
                for x, y, c in last_keypoints:
                    if c > 0.3:
                        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

            mode_text = f"Modo: {mode} | Gesto: {last_detected_gesture}"
            cv2.putText(
                vis, mode_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

            cv2.imshow("Gestos (webcam)", vis)

            # mostrar camera robobo con bbox si disponible
            if robo_frame is not None:
                vis2 = robo_frame.copy()
                if obj_visible and "box" in locals() and box is not None:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(
                        vis2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                    )
                    cv2.putText(
                        vis2,
                        f"obj size={obj_size:.3f}",
                        (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )

                cv2.imshow("Cam Robot / Obj", vis2)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        # cerrar fallback capture de robot
        cap_robo = getattr(read_robo_frame, "_cap", None)
        if cap_robo is not None:
            cap_robo.release()
        cv2.destroyAllWindows()
        if HAVE_ROBO:
            try:
                robobo.moveWheelsByTime(0, 0, 0.1)
            except Exception:
                pass


if __name__ == "__main__":
    main()

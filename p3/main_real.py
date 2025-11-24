import cv2
import time
import numpy as np

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

from gesture_control import run_pose_inference, detect_gesture_from_pose
import ar_control_sim as ar_sim

# -----------------------------
# CONFIG ROBOT / SIM
# -----------------------------
SIM_HOST = "localhost"

robobo = Robobo(SIM_HOST)
robobo.connect()
sim = RoboboSim(SIM_HOST)
sim.connect()

# Configurar cámara del Robobo (no bloqueante)
try:
    robobo.movePanTo(0, 50, wait=False)
    robobo.moveTiltTo(90, 50, wait=False)
    time.sleep(0.2)
except Exception:
    pass

# Activar blobs rojos como en P1
robobo.setActiveBlobs(True, False, False, False)

# Parámetros (coherentes con p1/env.py)
SPEED = 20.0  # env.speed
WHEELS_TIME = 0.25  # env.wheels_time

# Cámara PC para gestos
CAMERA_ID = 0
CONF_POSE = 0.5
FRAME_SKIP_POSE = 0

# Estado
current_command = "STOP"
last_detected_gesture = "STOP"
last_keypoints = None
mode = "GESTOS"  # "GESTOS" o "AR"
frame_count = 0


# ===================================================
# ACCIONES: GESTOS -> ACCIÓN CONTINUA, APLICACIÓN UNIFICADA
# ===================================================
def action_from_gesture(gesture: str) -> np.ndarray:
    if gesture == "FORWARD":
        return np.array([+0.6, +0.6], dtype=np.float32)
    elif gesture == "LEFT":
        return np.array([-0.5, +0.5], dtype=np.float32)
    elif gesture == "RIGHT":
        return np.array([+0.5, -0.5], dtype=np.float32)
    elif gesture == "STOP":
        return np.array([0.0, 0.0], dtype=np.float32)
    else:
        return np.array([0.0, 0.0], dtype=np.float32)


def apply_action(action: np.ndarray, source: str):
    """
    Usa la misma conversión que en p1/env.py.step:
      left_vel, right_vel in [-1,1] -> motors in [-100,100] via SPEED
      y se llama robobo.moveWheelsByTime(right_motor, left_motor, WHEELS_TIME)
    """
    action = np.asarray(action, dtype=np.float32).flatten()
    if action.size != 2:
        raise ValueError(f"Action must have 2 elements, received {action.size}")

    left_vel = float(np.clip(action[0], -1.0, 1.0))
    right_vel = float(np.clip(action[1], -1.0, 1.0))

    left_motor = int(np.clip(left_vel * SPEED, -100, 100))
    right_motor = int(np.clip(right_vel * SPEED, -100, 100))

    print(f"[{source}] action={np.round(action, 3)} -> L={left_motor}, R={right_motor}")
    robobo.moveWheelsByTime(right_motor, left_motor, WHEELS_TIME)


# ===================================================
# MAIN LOOP
# ===================================================
def reset_sim():
    try:
        sim.resetSimulation()
    except Exception:
        pass


def main():
    global current_command, last_detected_gesture, last_keypoints, mode, frame_count

    reset_sim()
    time.sleep(0.2)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("No se puede abrir la webcam")
        return

    # bajar resolución para subir FPS si quieres
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tele_actions_count = 0
    MIN_TELE_ACTIONS = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # 1) POSE (gestos)
            run_pose = (FRAME_SKIP_POSE == 0) or (
                frame_count % (FRAME_SKIP_POSE + 1) == 0
            )
            if run_pose:
                keypoints, _ = run_pose_inference(frame, conf=CONF_POSE, imgsz=320)
                last_keypoints = keypoints
                gesture = detect_gesture_from_pose(keypoints)
                if gesture is not None:
                    last_detected_gesture = gesture

            # 2) Blob desde simulador (p1-style)
            blob_info = ar_sim.get_blob_info_sim(robobo, sim)
            blob_centered = (blob_info["visible"] == 1.0) and (
                abs(blob_info["x"] - 50.0) < 15
            )

            # 3) Modo y control
            if mode == "GESTOS":
                action = action_from_gesture(last_detected_gesture)
                apply_action(action, source="GESTOS")
                tele_actions_count += 1

                if tele_actions_count >= MIN_TELE_ACTIONS and blob_centered:
                    # cambio a AR
                    mode = "AR"
                    print("CAMBIO DE MODO: ahora controla la política de refuerzo (P1)")
                    # opcional pequeña espera
                    time.sleep(0.1)
            else:
                obs_8d = ar_sim.build_obs_8d_from_blob(blob_info, sim)
                action = ar_sim.ar_policy_step(obs_8d)
                apply_action(action, source="AR")
                # ritmo similar a env.step
                time.sleep(0.05)

            # 4) Visual / debug
            annotated = frame.copy()
            if last_keypoints is not None:
                for x, y, c in last_keypoints:
                    if c > 0.3:
                        cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)

            cv2.putText(
                annotated,
                f"Modo: {mode} | Gesto: {last_detected_gesture}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Gestos (webcam)", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            robobo.moveWheelsByTime(0, 0, 0.1)
        except Exception:
            pass


if __name__ == "__main__":
    main()

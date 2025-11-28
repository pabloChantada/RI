import cv2
import time
import numpy as np

try:
    from robobopy.Robobo import Robobo
    from robobopy_videostream.RoboboVideo import RoboboVideo

    HAVE_ROBO = True
except Exception:
    HAVE_ROBO = False

from gesture_control import run_pose_inference, detect_gesture_from_pose
import ar_control_sim as ar_sim

# -----------------------------
# CONFIG ROBOT / SIM
# -----------------------------

WEBCAM_ID = 0  # PC webcam for gesture control
ROBO_IP = "10.56.43.50"  # smartphone IP (Robobo camera)

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

# Unified wheel scaling (similar to P1)
SPEED = 60.0
WHEELS_TIME = 0.20

# Stop condition based on bbox relative area (object "close enough")
STOP_SIZE_THRESHOLD = 0.30  # tune with experiments

# State
mode = "GESTOS"
last_detected_gesture = "STOP"
last_keypoints = None
mode = "GESTOS"  # "GESTOS" o "AR"
frame_count = 0

# ---------------------------------
# ROBOT (Robobo + RoboboVideo)
# ---------------------------------
robobo = None
videoStream = None

if HAVE_ROBO:
    robobo = Robobo(ROBO_IP)
    robobo.connect()

    videoStream = RoboboVideo(ROBO_IP)
    videoStream.connect()
    robobo.startStream()

    # Initial camera orientation
    try:
        robobo.movePanTo(0, 50, wait=False)
        robobo.moveTiltTo(100, 50, wait=False)
        time.sleep(0.2)
    except Exception:
        pass


# ===================================================
# Action mapping and application
# ===================================================
def action_from_gesture(gesture: str) -> np.ndarray:
    """Map discrete gesture to continuous wheel action [left, right] in [-1, 1]."""
    if gesture == "FORWARD":
        return np.array([+0.6, +0.6], dtype=np.float32)
    elif gesture == "LEFT":
        return np.array([-0.5, +0.5], dtype=np.float32)
    elif gesture == "RIGHT":
        return np.array([+0.5, -0.5], dtype=np.float32)
    elif gesture == "STOP":
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def apply_action(action: np.ndarray, source: str = "GESTOS"):
    """Scale continuous action to Robobo wheel commands and apply."""
    action = np.asarray(action, dtype=np.float32).flatten()
    if action.size != 2:
        raise ValueError(f"Action must have 2 elements, received {action.size}")

    left_vel = float(np.clip(action[0], -1.0, 1.0))
    right_vel = float(np.clip(action[1], -1.0, 1.0))

    left_motor = int(np.clip(left_vel * SPEED, -100, 100))
    right_motor = int(np.clip(right_vel * SPEED, -100, 100))

    print(f"[{source}] action={np.round(action, 3)} -> L={left_motor}, R={right_motor}")
    robobo.moveWheelsByTime(right_motor, left_motor, WHEELS_TIME)

    if HAVE_ROBO:
        robobo.moveWheelsByTime(right_motor, left_motor, WHEELS_TIME)


# ===================================================
# Robobo camera frame reading
# ===================================================
def read_robo_frame():
    """Read a BGR frame from the Robobo smartphone camera."""
    if HAVE_ROBO and videoStream is not None:
        try:
            frame, timestamp, sync_id, frame_id = videoStream.getImageWithMetadata()
            return frame
        except Exception:
            return None
    return None


# ===================================================
# MAIN
# ===================================================
def main():
    """Main control loop: teleop with gestures + AR policy handoff."""
    global mode, last_detected_gesture, last_keypoints, frame_count

    # PC webcam for gestures
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Could not open gesture webcam")
        return

    tele_actions_count = 0
    MIN_TELE_ACTIONS = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # --- Gesture detection (YOLO pose) ---
            run_pose = (FRAME_SKIP_POSE == 0) or (
                frame_count % (FRAME_SKIP_POSE + 1) == 0
            )
            if run_pose:
                keypoints, _ = run_pose_inference(frame, conf=CONF_POSE, imgsz=320)
                last_keypoints = keypoints
                gesture = detect_gesture_from_pose(keypoints)
                if gesture is not None:
                    last_detected_gesture = gesture

            # --- Object detection (Robobo camera + YOLO COCO 'bottle') ---
            robo_frame = read_robo_frame()
            obj_visible = False
            obj_x = obj_y = obj_size = 0.0
            box = None
            if robo_frame is not None:
                obj_visible, obj_x, obj_y, obj_size, _, box = (
                    ar_real.run_object_inference(robo_frame, conf=CONF_OBJ, imgsz=320)
                )

            centered = obj_visible and (0.4 <= obj_x <= 0.6)

            # 3) Modo y control
            if mode == "GESTOS":
                # Teleoperation: apply gesture-based action
                action = action_from_gesture(last_detected_gesture)
                apply_action(action, source="GESTOS")
                tele_actions_count += 1

                if tele_actions_count >= MIN_TELE_ACTIONS and blob_centered:
                    # cambio a AR
                    mode = "AR"
                    print("MODE SWITCH -> AR (RL policy active)")
                    time.sleep(0.1)
            else:
                # --- AR mode (RL policy from P1) ---

                # Stop criterion: object large enough in the image
                if obj_visible and obj_size >= STOP_SIZE_THRESHOLD:
                    print(
                        f"[AR] TARGET REACHED: size={obj_size:.3f} "
                        f">= STOP_TH={STOP_SIZE_THRESHOLD:.3f}. Stopping."
                    )
                    apply_action(
                        np.array([0.0, 0.0], dtype=np.float32), source="AR-STOP"
                    )
                    break

                # Build 8D observation and run SAC policy
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
                time.sleep(0.08)

            # --- Visual debug: gesture webcam ---
            vis = frame.copy()
            if last_keypoints is not None:
                for x, y, c in last_keypoints:
                    if c > 0.3:
                        cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)

            mode_text = f"Mode: {mode} | Gesture: {last_detected_gesture}"
            cv2.putText(
                annotated,
                f"Modo: {mode} | Gesto: {last_detected_gesture}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Gestures (webcam)", vis)

            # --- Visual debug: Robobo camera + bbox ---
            if robo_frame is not None:
                vis2 = robo_frame.copy()
                if obj_visible and box is not None:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(
                        vis2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                    )
                    cv2.putText(
                        vis2,
                        f"bottle size={obj_size:.3f}",
                        (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )
                    cv2.putText(
                        vis2,
                        f"STOP_TH={STOP_SIZE_THRESHOLD:.3f}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

                cv2.imshow("Robobo Cam / Object (bottle)", vis2)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if HAVE_ROBO:
            try:
                robobo.moveWheelsByTime(0, 0, 0.1)
            except Exception:
                pass
            try:
                videoStream.disconnect()
            except Exception:
                pass
            try:
                robobo.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()

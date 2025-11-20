import cv2
import numpy as np
from ultralytics import YOLO
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

MODEL_PATH = "yolo11n-pose.pt"
model = YOLO(MODEL_PATH)

sim_host = "localhost"
robobo = Robobo(sim_host)
robobo.connect()
sim = RoboboSim(sim_host)
sim.connect()

LEFT_SPEED = 15
RIGHT_SPEED = 15
FORWARD_SPEED = 30
MOVE_TIME = 0.1  # más pequeño para hacer el movimiento más fluido

CAMERA_ID = 0
CONF_THRESHOLD = 0.5

current_command = "STOP"  # comando actualmente aplicado al robot
last_detected_gesture = "STOP"
last_keypoints = None

FRAME_SKIP = (
    1  # procesa YOLO en 1 de cada (FRAME_SKIP+1) frames: 1 -> cada 2; 2 -> cada 3
)


def get_keypoint(kps, index):
    if kps is None or index >= len(kps):
        return None, None, 0
    return kps[index, 0], kps[index, 1], kps[index, 2]


def detect_gesture_from_pose(keypoints):
    """
    Gestos:
      - brazos abajo  -> STOP
      - dos brazos arriba -> FORWARD
      - solo brazo derecho arriba -> RIGHT
      - solo brazo izquierdo arriba -> LEFT
    """

    if keypoints is None:
        return None

    # hombros y muñecas
    l_sh_x, l_sh_y, c_lsh = get_keypoint(keypoints, 5)
    r_sh_x, r_sh_y, c_rsh = get_keypoint(keypoints, 6)
    l_wr_x, l_wr_y, c_lwr = get_keypoint(keypoints, 9)
    r_wr_x, r_wr_y, c_rwr = get_keypoint(keypoints, 10)

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
        rel_y_left = (l_sh_y - l_wr_y) / shoulder_dist  # >0 si muñeca por encima hombro
        left_up = rel_y_left > 0.4

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


def send_robot_command(command):
    # Aplica SIEMPRE el comando actual durante MOVE_TIME
    if command == "FORWARD":
        robobo.moveWheelsByTime(FORWARD_SPEED, FORWARD_SPEED, MOVE_TIME)
    elif command == "LEFT":
        robobo.moveWheelsByTime(LEFT_SPEED, -RIGHT_SPEED, MOVE_TIME)
    elif command == "RIGHT":
        robobo.moveWheelsByTime(-LEFT_SPEED, RIGHT_SPEED, MOVE_TIME)
    elif command == "STOP":
        robobo.moveWheelsByTime(0, 0, MOVE_TIME)


def main():
    global current_command, last_detected_gesture, last_keypoints

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    # Bajamos resolución de cámara para subir FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 520)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Ejecutamos YOLO solo cada (FRAME_SKIP+1) frames
        if frame_count % (FRAME_SKIP + 1) == 0:
            results = model(frame, conf=CONF_THRESHOLD, imgsz=320, verbose=False)

            keypoints = None
            if len(results) > 0 and len(results[0].keypoints) > 0:
                kp = results[0].keypoints[0]
                keypoints = np.concatenate(
                    [kp.xy[0].cpu().numpy(), kp.conf[0].cpu().numpy().reshape(-1, 1)],
                    axis=1,
                )

            last_keypoints = keypoints

            gesture = detect_gesture_from_pose(keypoints)
            if gesture is not None:
                current_command = gesture
                last_detected_gesture = gesture

        # En cada iteración mandamos SIEMPRE el comando actual al robot
        send_robot_command(current_command)

        # Debug visual: siempre mostramos el frame actual, con los últimos keypoints
        annotated = frame.copy()
        if last_keypoints is not None:
            for x, y, c in last_keypoints:
                if c > 0.3:
                    cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)

        cv2.putText(
            annotated,
            f"Gesto: {last_detected_gesture if last_detected_gesture else 'None'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("YOLO Pose Control", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    # por si acaso, parar el robot al salir
    robobo.moveWheelsByTime(0, 0, 0.1)


def reset():
    try:
        sim.resetSimulation()
    except Exception:
        pass


if __name__ == "__main__":
    reset()
    main()

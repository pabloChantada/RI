# main.py

import time

import cv2
import numpy as np

import config
from robobo_interface import RoboboInterface, setup_signal_handlers
from vision import VisionSystem
from teleop_controller import infer_gesture_from_keypoints, gesture_to_command
from rl_policy import load_policy


def teleop_and_rl_demo():
    """
    Flujo principal:
    1) Teleoperación por postura hasta detectar objeto y hacer >= MIN_TELEOP_ACTIONS
    2) Pasar a política de refuerzo para aproximarse al objeto

    """
    robot = RoboboInterface(config.ROBOT_IP)
    setup_signal_handlers(robot)

    robot.connect()
    vision = VisionSystem()
    policy = load_policy()

    mode = "TELEOP"
    teleop_actions = 0

    try:
        while True:
            frame = robot.read_frame()
            if frame is None:
                print("[Main] No se recibe frame de vídeo, esperando...")
                time.sleep(0.05)
                continue

            if mode == "TELEOP":
                keypoints = vision.detect_pose(frame)
                gesture = infer_gesture_from_keypoints(keypoints, frame.shape)
                command = gesture_to_command(gesture)

                # Solo contamos acciones “de verdad” (no STOP)
                if command != "STOP":
                    teleop_actions += 1

                print(f"[TELEOP] gesto={gesture} -> comando={command}")
                robot.execute_command(command)

                # Si ya hemos hecho X acciones, empezamos a buscar objeto
                if teleop_actions >= config.MIN_TELEOP_ACTIONS:
                    detection = vision.detect_target(frame)
                    if detection is not None:
                        bbox, conf = detection
                        print(f"[TELEOP] Objeto '{config.TARGET_CLASS_NAME}' detectado con conf={conf:.2f}")
                        print("[TELEOP] Cambiando a modo RL...")
                        robot.stop()
                        mode = "RL"
                        # no hacemos continue, dejamos que pase a RL en siguiente iteración

            else:  # mode == "RL"
                detection = vision.detect_target(frame)
                if detection is None:
                    # Estrategia simple de búsqueda si se pierde el objeto
                    print("[RL] Objeto perdido, girando para buscar...")
                    robot.turn_left()
                    continue

                bbox, conf = detection
                blob_size, blob_x, blob_y = vision.bbox_to_blob_metrics(bbox, frame.shape)
                obs = np.array([blob_size, blob_x, blob_y], dtype=np.float32)

                print(f"[RL] blob_size={blob_size:.3f}, x={blob_x:.3f}, y={blob_y:.3f}")

                # Condición de llegada
                if blob_size > config.TARGET_REACHED_BLOB_SIZE:
                    print("[RL] ¡Objeto alcanzado! blob_size por encima del umbral.")
                    robot.stop()
                    break

                action_idx = policy.predict(obs)
                robot.execute_policy_action(action_idx)

            # Opcional: muestra del frame (útil para debug, no obligatorio en la práctica)
            cv2.imshow("Robobo cam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC para salir
                print("[Main] ESC pulsado, saliendo...")
                break

    finally:
        robot.disconnect()


# Versión solo RL para usar desde test.py
def rl_only_demo():
    """
    Ejecuta sólo la parte de RL (sin teleoperación).
    Útil para el script test.py de la entrega.
    
    """
    robot = RoboboInterface(config.ROBOT_IP)
    setup_signal_handlers(robot)

    robot.connect()
    vision = VisionSystem()
    policy = load_policy()

    try:
        while True:
            frame = robot.read_frame()
            if frame is None:
                print("[RL_ONLY] No se recibe frame de vídeo, esperando...")
                time.sleep(0.05)
                continue

            detection = vision.detect_target(frame)
            if detection is None:
                print("[RL_ONLY] Objeto no detectado, girando para buscar...")
                robot.turn_left()
                continue

            bbox, conf = detection
            blob_size, blob_x, blob_y = vision.bbox_to_blob_metrics(bbox, frame.shape)
            obs = np.array([blob_size, blob_x, blob_y], dtype=np.float32)

            print(f"[RL_ONLY] blob_size={blob_size:.3f}, x={blob_x:.3f}, y={blob_y:.3f}")

            if blob_size > config.TARGET_REACHED_BLOB_SIZE:
                print("[RL_ONLY] ¡Objeto alcanzado! blob_size por encima del umbral.")
                robot.stop()
                break

            action_idx = policy.predict(obs)
            robot.execute_policy_action(action_idx)

            cv2.imshow("Robobo cam - RL only", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[RL_ONLY] ESC pulsado, saliendo...")
                break

    finally:
        robot.disconnect()


if __name__ == "__main__":
    teleop_and_rl_demo()

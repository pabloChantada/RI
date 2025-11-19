import signal
import sys
from typing import Optional, Tuple

import cv2
from robobopy.Robobo import Robobo
from robobopy_videostream.RoboboVideo import RoboboVideo

import config


class RoboboInterface:
    """
    Wrapper de Robobo + RoboboVideo.

    """

    def __init__(self, ip: str):
        self.ip = ip
        self.rob: Optional[Robobo] = None
        self.video: Optional[RoboboVideo] = None
        self._connected = False

    # ------------------ Conexión / limpieza ------------------

    def connect(self):
        print(f"[Robobo] Conectando a {self.ip}...")
        self.rob = Robobo(self.ip)
        self.video = RoboboVideo(self.ip)

        self.rob.connect()
        self.video.connect()

        # Cámara frontal y streaming
        self.rob.setFrontCamera()
        self.rob.startCamera()
        self.rob.setStreamFps(20)
        self.rob.setCameraFps(20)
        self.rob.startStream()

        self._connected = True
        print("[Robobo] Conexión establecida.")

    def disconnect(self):
        print("[Robobo] Desconectando...")
        try:
            if self.video is not None:
                self.video.disconnect()
        except Exception as e:
            print(f"[Robobo] Error al desconectar vídeo: {e}")

        try:
            if self.rob is not None:
                self.rob.stopMotors()
                self.rob.stopStream()
                self.rob.stopCamera()
                self.rob.disconnect()
        except Exception as e:
            print(f"[Robobo] Error al desconectar base: {e}")

        cv2.destroyAllWindows()
        self._connected = False
        print("[Robobo] Desconectado.")

    # ------------------ Cámara ------------------

    def read_frame(self) -> Optional[Tuple]:
        """
        Devuelve el frame actual de la cámara del smartphone (BGR).
        """
        if self.video is None:
            return None
        frame, _, _, _ = self.video.getImageWithMetadata()
        return frame

    # ------------------ Movimientos ------------------

    def move_forward(self):
        print("avanzar hacia adelante")
        self.rob.moveWheelsByTime(
            config.FORWARD_SPEED,
            config.FORWARD_SPEED,
            config.FORWARD_TIME,
        )

    def move_backward(self):
        print("retroceder")
        self.rob.moveWheelsByTime(
            -config.BACKWARD_SPEED,
            -config.BACKWARD_SPEED,
            config.BACKWARD_TIME,
        )

    def turn_left(self):
        print("giro izquierda")
        self.rob.moveWheelsByTime(
            -config.TURN_SPEED,
            config.TURN_SPEED,
            config.TURN_TIME,
        )

    def turn_right(self):
        print("giro derecha")
        self.rob.moveWheelsByTime(
            config.TURN_SPEED,
            -config.TURN_SPEED,
            config.TURN_TIME,
        )

    def stop(self):
        print("parar")
        self.rob.stopMotors()

    # ------------------ API de alto nivel para comandos string ------------------

    def execute_command(self, command: str):
        """
        Ejecuta comandos de alto nivel:
        "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP".
        """
        command = command.upper()
        if command == "FORWARD":
            self.move_forward()
        elif command == "BACKWARD":
            self.move_backward()
        elif command == "LEFT":
            self.turn_left()
        elif command == "RIGHT":
            self.turn_right()
        else:
            self.stop()

    # ------------------ API para acciones de la política ------------------

    def execute_policy_action(self, action_index: int):
        """
        Convierte un índice de acción en movimiento real.
        Mapa definido en config.RL_ACTION_MAPPING.
        """
        name = config.RL_ACTION_MAPPING.get(action_index, "STOP")
        print(f"ejecutar política de refuerzo: acción {action_index} ({name})")
        self.execute_command(name)


def setup_signal_handlers(robot: RoboboInterface):
    def handler(sig, frame):
        print("\n[Robobo] Señal recibida, cerrando...")
        robot.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

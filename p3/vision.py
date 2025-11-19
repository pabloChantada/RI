# vision.py

from typing import Optional, Tuple

import numpy as np
from ultralytics import YOLO

import config


class VisionSystem:
    """
    Wrapper de YOLO pose + YOLO detección de objetos.

    """

    def __init__(
        self,
        pose_model_path: str = config.POSE_MODEL_PATH,
        obj_model_path: str = config.OBJ_MODEL_PATH,
        target_class_name: str = config.TARGET_CLASS_NAME,
    ):
        print("[Vision] Cargando modelos YOLO...")
        self.pose_model = YOLO(pose_model_path)
        self.obj_model = YOLO(obj_model_path)
        self.target_class_name = target_class_name
        print("[Vision] Modelos YOLO cargados.")

    # ------------- Pose: devuelve keypoints de la primera persona -------------

    def detect_pose(self, frame) -> Optional[np.ndarray]:
        """
        Devuelve un array (17, 2) con los puntos clave de la primera persona,
        en coordenadas de imagen (x, y). Si no hay persona, devuelve None.
        """
        results = self.pose_model(frame, verbose=False)
        if not results:
            return None
        res = results[0]
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            return None

        # Primera persona detectada
        keypoints_xy = res.keypoints.xy[0].cpu().numpy()  # shape (17, 2)
        return keypoints_xy

    # ----------- Detección de objeto: devuelve bbox y confianza --------------

    def detect_target(
        self, frame
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detiene la clase objetivo en el frame.
        Devuelve (bbox_xyxy, conf) si lo encuentra, o None si no.
        bbox_xyxy: np.array([x1, y1, x2, y2])

        """
        results = self.obj_model(frame, verbose=False)
        if not results:
            return None

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None

        best_box = None
        best_conf = 0.0

        names = r.names  # dict: id -> name

        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, "")
            if cls_name != self.target_class_name:
                continue
            conf = float(box.conf.item())
            if conf > best_conf:
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                best_conf = conf
                best_box = xyxy

        if best_box is None:
            return None

        return best_box, best_conf

    # ------------------ Conversión bbox -> métricas blob ---------------------

    @staticmethod
    def bbox_to_blob_metrics(
        bbox_xyxy: np.ndarray, frame_shape
    ) -> Tuple[float, float, float]:
        """
        Convierte un bbox en (blob_size, blob_x, blob_y).

        - blob_size = área relativa (0..1)
        - blob_x = posición horizontal normalizada (-1..1, 0 centro)
        - blob_y = posición vertical normalizada (-1..1, 0 centro)

        """
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h
        blob_size = float(area) / float(w * h + 1e-6)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        blob_x = (cx - w / 2.0) / (w / 2.0)
        blob_y = (cy - h / 2.0) / (h / 2.0)

        return float(blob_size), float(blob_x), float(blob_y)

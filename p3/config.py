ROBOT_IP = "localhost"  

### Modelos YOLO ###

# Rutas a los modelos de Ultralytics YOLO
POSE_MODEL_PATH = "yolo11n-pose.pt"   # o "yolov8n-pose.pt"
OBJ_MODEL_PATH = "yolo11n.pt"         # o "yolov8n.pt"

# Clase objetivo que YOLO debe detectar (por ejemplo, bottle, cup, etc.)

TARGET_CLASS_NAME = "bottle"          # hay que cambiar esto según nuestro objeto ofc

### Teleoperación ###

# Número mínimo de acciones de teleop antes de pasar a RL
MIN_TELEOP_ACTIONS = 5

### Política de refuerzo ###

# Sin RL hay que poner True (heurística)
# Para usar la política real de la práctica 1, se pone False y hay que completar rl_policy.py
USE_HEURISTIC_POLICY = True

# Ruta a tu Q-table o modelo de la práctica 1
QTABLE_PATH = "ruta_a_cambiar_chantada"   

# Bins de discretización para blob_size, x, y.
# AJUSTAR ESTOS VALORES SEGÚN LA P1

SIZE_BINS = [0.05, 0.15, 0.3, 0.6, 1.0]
X_BINS    = [-0.75, -0.25, 0.25, 0.75, 1.0]
Y_BINS    = [-0.75, -0.25, 0.25, 0.75, 1.0]

# Número de acciones de la política
NUM_ACTIONS = 4

# Mapeo índice de acción (comando de movimiento de alto nivel)

RL_ACTION_MAPPING = {
    0: "FORWARD",   
    1: "LEFT",      
    2: "RIGHT",     
    3: "STOP",      
}

### Condición de “objeto alcanzado” ###

TARGET_REACHED_BLOB_SIZE = 0.5

### Parámetros de movimiento ###

FORWARD_SPEED = 40
TURN_SPEED    = 30
BACKWARD_SPEED = 35

FORWARD_TIME = 0.3
TURN_TIME    = 0.25
BACKWARD_TIME = 0.3

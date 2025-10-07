# En la primera iteracion unicamente tiene que encontrar el blob rojo y moverse hacia el

# Hay que usar StableBaselines3, con enviroments que sigan las estructura de Gymnasium
# Para la creacion del env, es necesario:
# - Crear una clase que herede de gym.Env (https://gymnasium.farama.org/introduction/create_custom_env/)
# - A esta clase hay que implementarle los metodos:
#   - Espacio de observaciones/estados
#   - Espacio de acciones (cualquier accion aleatoria)
#   - Funcion de recompensa
#   - Politica del algoritmo

# Es necesario presentar resultados de las metricas: "mean_reward", "ep_reward_mean". 
# Usando otras librerias como seaborn para resultados en formato .png, .jpeg, etc.
# Tambien es necesario guardar un plano 2D de las diferentes posiciones que a realizado el agente

# Si la recompensa durante X acciones es negativa, seleccionar una nueva accion aleatoria 
# o resetear el entorno

from typing import Optional
import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo 
from robobosim.RoboboSim import RoboboSim
import time
from robobopy.utils.BlobColor import BlobColor
import random

class CustomEnv(gym.Env):
    """
    Entorno personalizado para entrenar un robot Robobo a localizar
    y acercarse a un cilindro rojo usando visión por computador.
    
    Observation Space:
        - Posición del agente (x, z) normalizada: [-1, 1]
        - Posición del objetivo (x, z) normalizada: [-1, 1]
        - Blob visible: {0, 1}
        - Posición del blob (x, y) normalizada: [-1, 1]
        - Tamaño del blob normalizado: [0, 1]
    
    Action Space:
        - Velocidades continuas de las ruedas: [-1, 1] para cada rueda
    """
    
    def __init__(self, size: int = 1000, max_steps: int = 30):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Umbral para considerar que se alcanzó el objetivo
        self.goal_threshold = 150.0
        
        # Frecuencia de movimiento del target (cada N pasos)
        self.target_move_frequency = 2 

        # Conectar al simulador Robobo
        self.robobo = Robobo("localhost")
        self.robobo.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()
        
        # Configurar detección de blobs (solo RED)
        self.robobo.setActiveBlobs(True, False, False, False)
        
        # Ajustar posición de la cámara
        self._setup_camera()
        
        # Obtener ID del objeto objetivo (cilindro)
        self._object_id = list(self.sim.getObjects())[0]
        
        # OBSERVATION SPACE: Todos los valores normalizados
        # [agent_x, agent_z, target_x, target_z, 
        #  blob_visible, blob_x, blob_y, blob_size]
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -1.0, -1.0,  # agent position
                -1.0, -1.0,  # target position
                0.0,         # blob_visible (0 o 1)
                -1.0, -1.0,  # blob position (centrado en 0)
                0.0          # blob_size
            ], dtype=np.float32),
            high=np.array([
                1.0, 1.0,    # agent position
                1.0, 1.0,    # target position
                1.0,         # blob_visible
                1.0, 1.0,    # blob position
                1.0          # blob_size
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # ACTION SPACE: Velocidades continuas de las ruedas
        # [left_wheel_velocity, right_wheel_velocity] en [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Parámetros para conversión de acciones
        self._motor_scale = 20.0      # Escala para convertir [-1,1] a velocidad del motor
        self._cmd_duration = 0.25     # Duración del comando en segundos
        
        # Variables para seguimiento
        self.previous_distance = None
        self.initial_distance = None

    def _setup_camera(self):
        """Configura la posición de la cámara del robot"""
        self.robobo.movePanTo(0, 50)
        self.robobo.moveTiltTo(105, 50)
        time.sleep(0.5)

    def _get_blob_info(self):
        """
        Obtiene información del blob (cilindro) detectado por la cámara
        
        Returns:
            dict: Información del blob con claves 'visible', 'x', 'y', 'size'
        """
        try:
            blob = self.robobo.readColorBlob(BlobColor.RED)
            agent_x, agent_z = self._get_agent_position()
            target_x, target_z = self._get_object_position()
            
            distance_to_target = np.sqrt((agent_x - target_x)**2 + (agent_z - target_z)**2)

            # Caso 1: No se detecta blob
            if blob is None or blob.size <= 0:
                return {
                    "visible": 0.0,
                    "x": 50.0,      # Centro de la imagen
                    "y": 50.0,      # Centro de la imagen
                    "size": 0.0
                }
            
            # Caso 2: Estamos muy cerca del objetivo (alcanzado)
            elif distance_to_target < self.goal_threshold:
                return {
                    "visible": 1.0,
                    "x": 50.0,      # Centro (objetivo centrado)
                    "y": 50.0,      # Centro
                    "size": 400.0   # Tamaño máximo
                }
            
            # Caso 3: Blob detectado normalmente
            else:
                return {
                    "visible": 1.0,
                    "x": float(blob.posx),      # Rango [0-100]
                    "y": float(blob.posy),      # Rango [0-100]
                    "size": float(blob.size)    # Rango [0-100+]
                }
                
        except Exception as e:
            print(f"Error leyendo blob: {e}")
            return {
                "visible": 0.0,
                "x": 50.0,
                "y": 50.0,
                "size": 0.0
            }

    def _get_agent_position(self):
        """Obtiene la posición actual del robot"""
        try:
            agent_location = self.sim.getRobotLocation(0)
            return agent_location["position"]["x"], agent_location["position"]["z"]
        except Exception as e:
            print(f"Error obteniendo posición del agente: {e}")
            return 0.0, 0.0

    def _get_object_position(self):
        """Obtiene la posición actual del objeto objetivo"""
        try:
            target_location = self.sim.getObjectLocation(self._object_id)
            return target_location["position"]["x"], target_location["position"]["z"]
        except Exception as e:
            print(f"Error obteniendo posición del objeto: {e}")
            return 0.0, 0.0

    def _get_obs(self):
        """
        Genera la observación actual del entorno (NORMALIZADA)
        
        Returns:
            np.ndarray: Vector de observación normalizado
        """
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        blob_info = self._get_blob_info()
        
        # Normalizar posiciones: [-1000, 1000] -> [-1, 1]
        agent_x_norm = np.clip(agent_x / 1000.0, -1.0, 1.0)
        agent_z_norm = np.clip(agent_z / 1000.0, -1.0, 1.0)
        target_x_norm = np.clip(target_x / 1000.0, -1.0, 1.0)
        target_z_norm = np.clip(target_z / 1000.0, -1.0, 1.0)
        
        # Normalizar posición del blob: [0, 100] -> [-1, 1] (centrado en 50)
        blob_x_norm = np.clip((blob_info["x"] - 50.0) / 50.0, -1.0, 1.0)
        blob_y_norm = np.clip((blob_info["y"] - 50.0) / 50.0, -1.0, 1.0)
        
        # Normalizar tamaño del blob: [0, 400] -> [0, 1]
        blob_size_norm = np.clip(blob_info["size"] / 400.0, 0.0, 1.0)
        
        return np.array([
            agent_x_norm, agent_z_norm,           
            target_x_norm, target_z_norm,          
            blob_info["visible"],       
            blob_x_norm,             
            blob_y_norm,             
            blob_size_norm            
        ], dtype=np.float32)

    def _get_info(self):
        """
        Genera información auxiliar para debugging
        
        Returns:
            dict: Información del estado actual
        """
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        blob_info = self._get_blob_info()
        
        distance = np.sqrt((agent_x - target_x)**2 + (agent_z - target_z)**2)
        
        return {
            "distance": distance,
            "agent_position": (agent_x, agent_z),
            "target_position": (target_x, target_z),
            "step": self.current_step,
            "blob_visible": bool(blob_info["visible"]),
            "blob_centered": self._is_blob_centered(blob_info),
            "blob_size": blob_info["size"]
        }

    def _is_blob_centered(self, blob_info, margin=15):
        """
        Verifica si el blob está centrado en la vista de la cámara
        
        Args:
            blob_info: Información del blob
            margin: Margen de tolerancia en pixeles
        """
        if blob_info["visible"] == 0.0:
            return False
        
        center_x = 50.0
        return abs(blob_info["x"] - center_x) < margin
    
    def _randomize_target_position(self):
        """
        Coloca el objetivo en una posición aleatoria al inicio del episodio
        """
        try:
            target_location = self.sim.getObjectLocation(self._object_id)
            target_y = float(target_location["position"]["y"])
            
            # Generar posición aleatoria dentro del rango válido
            new_x = float(random.uniform(-800.0, 800.0))
            new_z = float(random.uniform(-800.0, 800.0))
            
            new_position = {
                "x": new_x,
                "y": target_y,
                "z": new_z
            }
            new_rotation = target_location.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
            
            self.sim.setObjectLocation(self._object_id, new_position, new_rotation)
            time.sleep(0.3)  # Esperar a que se actualice
            
        except Exception as e:
            print(f"Error randomizando posición del objetivo: {e}")

    def _move_target(self):
        """
        Mueve el objetivo a una nueva posición cercana (para target móvil)
        """
        try:
            target_location = self.sim.getObjectLocation(self._object_id)
            
            target_x = float(target_location["position"]["x"])
            target_y = float(target_location["position"]["y"])
            target_z = float(target_location["position"]["z"])
            
            # Movimiento aleatorio en x y z
            new_x = target_x + random.choice([-40.0, 40.0])
            new_z = target_z + random.choice([-40.0, 40.0])
            
            # Mantener dentro de los límites del entorno
            new_x = float(np.clip(new_x, -1000.0, 1000.0))
            new_z = float(np.clip(new_z, -1000.0, 1000.0))
            
            new_position = {
                "x": new_x,
                "y": target_y,
                "z": new_z
            }
            new_rotation = target_location.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
            
            self.sim.setObjectLocation(self._object_id, new_position, new_rotation)
            time.sleep(0.3)
            
            print(f"  [Target movido de ({target_x:.1f}, {target_z:.1f}) a ({new_x:.1f}, {new_z:.1f})]")
            
        except Exception as e:
            print(f"Error moviendo target: {e}")

    def _calculate_reward(self, current_distance, blob_info):
        """
        Calcula la recompensa basada en distancia y visión
        
        Estrategia de recompensa:
        1. Recompensa por acercarse al objetivo
        2. Bonificación por mantener el blob visible y centrado
        3. Penalización por perder de vista el blob
        4. Recompensa grande por alcanzar el objetivo
        
        Args:
            current_distance: Distancia actual al objetivo
            blob_info: Información del blob detectado
        
        Returns:
            float: Recompensa total
        """
        reward = 0.0
        
        # 1. Recompensa por reducir distancia
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - current_distance
            reward += distance_improvement * 0.01  # Escalar apropiadamente
        
        # 2. Recompensa por visibilidad del blob
        if blob_info["visible"] == 1.0:
            reward += 0.1
            
            # Bonificación por centrar el blob
            center_x = 50.0
            blob_x = blob_info["x"]
            distance_from_center = abs(blob_x - center_x)
            centering_reward = 0.2 * (1.0 - min(distance_from_center / 50.0, 1.0))
            reward += centering_reward
            
            # Bonificación por tamaño del blob (más cerca = más grande)
            size_reward = min(blob_info["size"] / 400.0, 1.0) * 0.1
            reward += size_reward
        else:
            # Penalización por perder el blob
            reward -= 0.3
        
        # 3. Recompensa por alcanzar el objetivo
        if current_distance < self.goal_threshold:
            reward += 5.0
        
        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reinicia el entorno para un nuevo episodio
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales
        
        Returns:
            observation: Observación inicial
            info: Información inicial
        """
        super().reset(seed=seed)
        
        # Resetear simulación
        self.sim.resetSimulation()
        time.sleep(0.8)
        
        # Randomizar posición del objetivo
        self._randomize_target_position()
        
        # Resetear detección de blobs
        self.robobo.resetColorBlobs()
        
        # Reconfigurar cámara
        self._setup_camera()
        
        # Resetear contadores
        self.current_step = 0
        info = self._get_info()
        self.initial_distance = info["distance"]
        self.previous_distance = self.initial_distance
        
        observation = self._get_obs()
        
        print(f"\n{'='*60}")
        print(f"NUEVO EPISODIO")
        print(f"Posición inicial agente: {info['agent_position']}")
        print(f"Posición objetivo: {info['target_position']}")
        print(f"Distancia inicial: {self.initial_distance:.1f}")
        print(f"{'='*60}\n")
        
        return observation, info

    def step(self, action):
        """
        Ejecuta una acción en el entorno
        
        Args:
            action: Array de 2 elementos [left_wheel_vel, right_wheel_vel] en [-1, 1]
        
        Returns:
            observation: Nueva observación
            reward: Recompensa obtenida
            terminated: Si el episodio terminó (objetivo alcanzado)
            truncated: Si el episodio fue truncado (max_steps alcanzado)
            info: Información adicional
        """
        self.current_step += 1

        # Mover el target cada N pasos
        if self.current_step % self.target_move_frequency == 0:
            print(f"\n  🎯 Moviendo target (step {self.current_step})...")
            self._move_target()
            # IMPORTANTE: Actualizar la distancia previa después de mover el objetivo
            # para que la recompensa no penalice injustamente al agente
            info_temp = self._get_info()
            self.previous_distance = info_temp["distance"]

        # Validar y procesar acción
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.size != 2:
            raise ValueError(f"La acción debe tener 2 elementos, se recibieron {action.size}")
        
        left_vel = float(np.clip(action[0], -1.0, 1.0))
        right_vel = float(np.clip(action[1], -1.0, 1.0))

        # Ejecutar acción en el simulador
        try:
            left_motor = int(np.clip(left_vel * self._motor_scale, -100, 100))
            right_motor = int(np.clip(right_vel * self._motor_scale, -100, 100))
            
            self.robobo.moveWheelsByTime(left_motor, right_motor, self._cmd_duration)
            time.sleep(0.5)  # Esperar a que se complete la acción
            
        except Exception as e:
            print(f"Error ejecutando acción: {e}")
        
        # Obtener nuevo estado
        observation = self._get_obs()
        info = self._get_info()
        current_distance = info["distance"]
        blob_info = self._get_blob_info()
        
        # Calcular recompensa
        reward = self._calculate_reward(current_distance, blob_info)
        
        # Verificar condiciones de terminación
        terminated = current_distance < self.goal_threshold
        truncated = self.current_step >= self.max_steps
        
        # Logging
        print(f"Step {self.current_step}/{self.max_steps} | "
              f"Acción: L={left_vel:.2f} R={right_vel:.2f} | "
              f"Dist: {current_distance:.1f} | "
              f"Blob: {'✓' if blob_info['visible'] else '✗'} | "
              f"Reward: {reward:.2f}")
        
        # Actualizar distancia previa
        self.previous_distance = current_distance
        
        return observation, reward, terminated, truncated, info

    def close(self):
        """Limpia recursos al cerrar el entorno"""
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
            print("Conexiones cerradas correctamente")
        except Exception as e:
            print(f"Error cerrando conexiones: {e}")
        super().close()

    def render(self):
        """Renderiza el entorno (opcional, no implementado)"""
        pass


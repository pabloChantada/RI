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
    Entorno cuadrado de busqueda de un cilindro ROJO. Usando como
    estructura base los entornos de Gymnasium

    Observation Space:
        - Posición del agente (x, z) normalizada: [-1, 1]
        - Posición del objetivo (x, z) normalizada: [-1, 1]
        - Blob visible: {0, 1} -> int
        - Tamaño del blob normalizado: [0, 1]
    
    Action Space:
        - Velocidades continuas de las ruedas: [-1, 1] para cada rueda
    """
    
    def __init__(self, size: int = 1000, max_steps: int = 30):
        super().__init__()

        self.size = size  # Indica como de "grande" se ve el blob en la camara
        self.max_steps = max_steps
        self.current_step = 0
        
        # Umbral para considerar que se alcanzó el objetivo
        self.goal_threshold = 150.0
        
        # Frecuencia de movimiento del target (cada N pasos)
        self.target_move_frequency = 5 

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
        self.speed = 20.0      # Escala para convertir [-1,1] a velocidad del motor
        self.wheels_time = 0.25  # Tiempo para realizar las acciones 
        # Variables para seguimiento
        self.previous_distance = None
        self.initial_distance = None

    def _setup_camera(self):
        """Configura la posición de la cámara del robot"""
        self.robobo.movePanTo(0, 50)  # Resetear el Pan por si acaso
        self.robobo.moveTiltTo(105, 50)  # (Grados, Speed) -> (Max hacia abajo, _)
        time.sleep(0.2)

    def _get_blob_info(self):
        """
        Obtiene información del blob (cilindro) detectado por la cámara
        
        Returns:
            dict: Información del blob con claves 'visible', 'x', 'y', 'size'
        """
        try:
            blob = self.robobo.readColorBlob(BlobColor.RED)
            # Las rotaciones no son necesarias exceptuando cuando lo requieren las funciones
            agent_x, agent_z = self._get_agent_position()
            target_x, target_z = self._get_object_position()
            
            # Usamos la euclidea de forma arbitraria realmente, aunque seria interesante usar la Manhattan )?
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
            agent_location = self.sim.getRobotLocation(0)  # Devuelve un trio (x,y,z); pero el eje y no se modifica nunca
            return agent_location["position"]["x"], agent_location["position"]["z"]
        except Exception as e:
            print(f"Error obteniendo posición del agente: {e}")
            return 0.0, 0.0

    def _get_object_position(self):
        """Obtiene la posición actual del objeto objetivo"""
        try:
            target_location = self.sim.getObjectLocation(self._object_id)  # Devuelve un trio (x,y,z); pero el eje y no se modifica nunca
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
        
        # Normalizar posición del blob respecto al centro de la camara: [0, 100] -> [-1, 1] (centrado en 50)
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
    
    def _move_target(self):
        """
        Mueve el objetivo en una trayectoria CURVA (como en tu dibujo)
        El cilindro se mueve hacia adelante/atrás (Z) Y hacia los lados (X) simultáneamente
        creando una curva suave hacia la izquierda o derecha
        """
        try:
            target_location = self.sim.getObjectLocation(self._object_id)
            
            target_x = float(target_location["position"]["x"])
            target_y = float(target_location["position"]["y"])
            target_z = float(target_location["position"]["z"])
            
            # INICIALIZACIÓN DE MOVIMIENTO CURVO
            
            if not hasattr(self, 'target_direction'):
                # Elegir dirección de la curva: -1 (curva izquierda) o +1 (curva derecha)
                self.target_direction = random.choice([-1, 1])
                self.target_speed_x = random.uniform(4.0, 6.0)  # Velocidad lateral
                self.target_speed_z = random.uniform(3.0, 5.0)  # Velocidad hacia adelante/atrás
                
                # Elegir si va hacia el robot (negativo) o se aleja (positivo)
                self.z_direction = random.choice([-1, 1])
                
                direction_name = "DERECHA" if self.target_direction > 0 else "IZQUIERDA"
                z_direction_name = "HACIA ROBOT" if self.z_direction < 0 else "ALEJÁNDOSE"
                print(f"\n  [Cilindro inicializado - Curva hacia {direction_name}, {z_direction_name}]")
                print(f"      Velocidad X: {self.target_speed_x:.1f}, Velocidad Z: {self.target_speed_z:.1f}")
            
            # MOVIMIENTO CURVO SIMULTÁNEO EN X y Z
            
            move_x = self.target_direction * self.target_speed_x  # Movimiento lateral
            move_z = self.z_direction * self.target_speed_z        # Movimiento adelante/atrás
            
            # Calcular nueva posición
            new_x = target_x + move_x
            new_z = target_z + move_z
            
            # DETECCIÓN DE BORDES Y CAMBIO DE DIRECCIÓN
            
            MARGIN_X = 150.0  # Margen lateral
            MARGIN_Z = 150.0  # Margen frontal/trasero
            changed_direction = False
            
            # Rebote en bordes laterales (X)
            if new_x <= -1000.0 + MARGIN_X:
                new_x = -1000.0 + MARGIN_X
                self.target_direction = 1  # Ahora curva hacia la derecha
                changed_direction = True
                print(f"\n  [🔄 Rebote LATERAL IZQUIERDO - Ahora curva hacia DERECHA]")
            elif new_x >= 1000.0 - MARGIN_X:
                new_x = 1000.0 - MARGIN_X
                self.target_direction = -1  # Ahora curva hacia la izquierda
                changed_direction = True
                print(f"\n  [🔄 Rebote LATERAL DERECHO - Ahora curva hacia IZQUIERDA]")
            
            # Rebote en bordes frontales/traseros (Z)
            if new_z <= -1000.0 + MARGIN_Z:
                new_z = -1000.0 + MARGIN_Z
                self.z_direction = 1  # Ahora se aleja del robot
                changed_direction = True
                print(f"\n  [🔄 Rebote FRONTAL - Ahora se ALEJA]")
            elif new_z >= 1000.0 - MARGIN_Z:
                new_z = 1000.0 - MARGIN_Z
                self.z_direction = -1  # Ahora va hacia el robot
                changed_direction = True
                print(f"\n  [🔄 Rebote TRASERO - Ahora va HACIA ROBOT]")
            
            # Asegurar que los valores estén en rango
            new_x = float(np.clip(new_x, -1000.0, 1000.0))
            new_z = float(np.clip(new_z, -1000.0, 1000.0))
            
            new_position = {
                "x": new_x,
                "y": target_y,
                "z": new_z
            }
            
            new_rotation = target_location.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
            
            # Mover el objeto
            self.sim.setObjectLocation(self._object_id, new_position, new_rotation)
            
            # Logging periódico
            # INICIALIZACIÓN DE MOVIMIENTO CURVO
            if not hasattr(self, 'target_direction'):
                # 🔹 Solo elegimos si la curva es hacia IZQUIERDA (-1) o DERECHA (+1)
                self.target_direction = random.choice([-1, 1])
                
                # 🔹 Siempre irá hacia ATRÁS (Z negativa)
                self.z_direction = -1

                # Velocidades base
                self.target_speed_x = random.uniform(4.0, 6.0)  # lateral
                self.target_speed_z = random.uniform(3.0, 5.0)  # profundidad

                direction_name = "DERECHA" if self.target_direction > 0 else "IZQUIERDA"
                print(f"\n  [Cilindro inicializado - Curva hacia atrás y a {direction_name}]")
                print(f"      Velocidad X: {self.target_speed_x:.1f}, Velocidad Z: {self.target_speed_z:.1f}")

            # MOVIMIENTO CURVO SIEMPRE HACIA ATRÁS
            move_x = self.target_direction * self.target_speed_x  # lateral
            move_z = -abs(self.target_speed_z)                    # siempre negativo (hacia atrás)

            
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
            # Usamos el 0.01 para evitar valores altos, de esta forma independientemente
            # del valor de "distance_improvement"; obtendremos un valor bajo (normalmentee entre [-1,1]
            reward += distance_improvement * 0.05 #CAMBIO de 0.1 a 0.05
        
        # 2. Recompensa por visibilidad del blob
        if blob_info["visible"] == 1.0:
            reward += 0.2 #CAMBIO de 0.1 a 0.2
            
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
            reward -= 0.15 # CAMBIO de 0.3 a 0.15
        
        # 3. Recompensa por alcanzar el objetivo
        if current_distance < self.goal_threshold:
            reward += 10.0 # CAMBIO de 5.0 a 10.0
        
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
        time.sleep(0.5)
        
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
        
        # print(f"\n{'='*60}")
        # print(f"NUEVO EPISODIO")
        # print(f"Posición inicial agente: {info['agent_position']}")
        # print(f"Posición objetivo: {info['target_position']}")
        # print(f"Distancia inicial: {self.initial_distance:.1f}")
        # print(f"{'='*60}\n")
        
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

        # Mover el target en CADA paso (movimiento continuo y fluido)
        # Solo si target_move_frequency es bajo (para fase 2)
        if hasattr(self, 'target_move_frequency') and self.target_move_frequency < 100:
            self._move_target()
            # Actualizamos la distancia previa después de mover el objetivo
            # para que la recompensa no penalice al agente
            info_temp = self._get_info()
            self.previous_distance = info_temp["distance"]

        # Validar y procesar acción
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.size != 2:
            raise ValueError(f"La acción debe tener 2 elementos, se recibieron {action.size}")
        

        # ESTO CREO QUE MEJOR LOS EXPLIQUE LAURA
        # DE AQUI ->
        left_vel = float(np.clip(action[0], -1.0, 1.0))
        right_vel = float(np.clip(action[1], -1.0, 1.0))

        # Ejecutar acción en el simulador
        try:
            left_motor = int(np.clip(left_vel * self.speed, -100, 100))
            right_motor = int(np.clip(right_vel * self.speed, -100, 100))
            
            self.robobo.moveWheelsByTime(right_motor, left_motor, self.wheels_time)
            time.sleep(0.25)  # Esperar a que se complete la acción
            
        except Exception as e:
            print(f"Error ejecutando acción: {e}")
        # <- HASTA AQUI

        # Obtener nuevo estado
        observation = self._get_obs()
        info = self._get_info()
        current_distance = info["distance"]
        blob_info = self._get_blob_info()
        
        # Calcular recompensa
        reward = self._calculate_reward(current_distance, blob_info)
        
        # Verificar condiciones de terminación
        terminated = current_distance < self.goal_threshold  # Fin de episodio por llegar a meta
        truncated = self.current_step >= self.max_steps      # Fin de episodio por tiempo
        
        # Logging
        print(f"Step {self.current_step}/{self.max_steps} | "
              f"Acción: L={left_vel:.2f} R={right_vel:.2f} | "
              f"Dist: {current_distance:.1f} | "
              f"Blob: {'1' if blob_info['visible'] else '0'} | "
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
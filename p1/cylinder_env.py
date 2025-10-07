# En la primera iteracion unicamente tiene que encontrar el blob rojo y moverse hacia el
#
# Hay que usar StableBaselines3, con enviroments que sigan las estructura de Gymnasium
# Para la creacion del env, es necesario:
# - Crear una clase que herede de gym.Env (https://gymnasium.farama.org/introduction/create_custom_env/)
# - A esta clase hay que implementarle los metodos:
#   - Espacio de observaciones/estados
#   - Espacio de acciones (cualquier accion aleatoria)
#   - Funcion de recompensa
#   - Politica del algoritmo
#
# Es necesario presentar resultados de las metricas: "mean_reward", "ep_reward_mean". 
# Usando otras librerias como seaborn para resultados en formato .png, .jpeg, etc.
# Tambien es necesario guardar un plano 2D de las diferentes posiciones que a realizado el agente
#
# Si la recompensa durante X acciones es negativa, seleccionar una nueva accion aleatoria 
# o resetear el entorno

from typing import Optional
import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo 
from robobosim.RoboboSim import RoboboSim
import time

class CylinderEnv(gym.Env):
    def __init__(self, size: int = 1000, max_steps: int = 30):
        super().__init__()

        # The size of the square grid (1000x1000 by default)
        self.size = size
        self.max_steps = max_steps  # Maximum steps per episode
        self.current_step = 0

        # Connect to Robobo simulator
        self.robobo = Robobo("localhost")
        self.robobo.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()
        
        # Get initial positions (there's always one robot and one object)
        self._object_id = list(self.sim.getObjects())[0] 
        
        # Observation space (sin cambios): [agent_x, agent_z, target_x, target_z]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1000.0, -1000.0, -1000.0, -1000.0], dtype=np.float32),
            high=np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # === CAMBIO 1: acciones CONTINUAS en [-1, 1] para (vl, vr) ===
        # Antes: Discrete(4). Ahora: Box(2) -> velocidades normalizadas de las dos ruedas.
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Escala y duración para mapear [-1,1] -> motores Robobo
        self._motor_scale = 30.0  # ajusta si quieres más/menos velocidad (rango aprox. de -100 a 100)
        self._cmd_duration = 0.5  # segundos por acción (igual que antes)
        
        # Initialize positions
        self.reset()

    def _get_agent_position(self):
        """Get current robot position"""
        try:
            agent_location = self.sim.getRobotLocation(0)
            return agent_location["position"]["x"], agent_location["position"]["z"]
        except Exception as e:
            print(f"Error getting agent position: {e}")
            return 0.0, 0.0

    def _get_object_position(self):
        """Get current target object position"""
        try:
            target_location = self.sim.getObjectLocation(self._object_id)
            return target_location["position"]["x"], target_location["position"]["z"]
        except Exception as e:
            print(f"Error getting object position: {e}")
            return 0.0, 0.0

    def _get_obs(self):
        """Convert internal state to observation format"""
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        return np.array([agent_x, agent_z, target_x, target_z], dtype=np.float32)

    def _get_info(self):
        """Compute auxiliary information for debugging"""
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        distance = np.sqrt((agent_x - target_x)**2 + (agent_z - target_z)**2)
        return {
            "distance": distance,
            "agent_position": (agent_x, agent_z),
            "target_position": (target_x, target_z),
            "step": self.current_step
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode"""
        super().reset(seed=seed)
        self.sim.resetSimulation()
        time.sleep(1) 
        self.current_step = 0
        self.initial_distance = self._get_info()["distance"]
        self.previous_distance = self.initial_distance
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Execute one timestep within the environment (ACCIONES CONTINUAS)"""
        self.current_step += 1
        print(f"\nStep {self.current_step}/{self.max_steps}")

        # === CAMBIO 2: interpretar acción continua [vl, vr] ∈ [-1,1] ===
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 2:
            raise ValueError(f"Action must have shape (2,), got {a.shape}")
        vl = float(np.clip(a[0], -1.0, 1.0))
        vr = float(np.clip(a[1], -1.0, 1.0))

        # Para ver en consola qué hace:
        print(f"Action (continuous): vl={vl:.3f}, vr={vr:.3f}")

        # Mapear a velocidades de motor Robobo y ejecutar durante _cmd_duration
        try:
            left = int(np.clip(vl * self._motor_scale, -100, 100))
            right = int(np.clip(vr * self._motor_scale, -100, 100))
            self.robobo.moveWheelsByTime(left, right, self._cmd_duration)
            time.sleep(self._cmd_duration + 0.1)
        except Exception as e:
            print(f"Error executing action: {e}")

        # Nuevo estado y métricas
        observation = self._get_obs()
        info = self._get_info()
        current_distance = info["distance"]
        print(f"Robot position: {info['agent_position']}, Target position: {info['target_position']}, Distance: {current_distance:.3f}")
        
        # Recompensa (sin cambios)
        reward = self._calculate_reward(current_distance)
        print(f"---> Reward: {reward:.3f}")

        # Terminación (sin cambios)
        terminated = current_distance < 150
        truncated = self.current_step >= self.max_steps
        
        self.previous_distance = current_distance
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, current_distance):
        """Calculate normalized reward based on distance to target"""
        distance_reward = (self.initial_distance - current_distance) * 10
        if current_distance < 10:
            return 1.0
        distance_penalty = -0.5 if current_distance > self.previous_distance else 0.0
        total_reward = distance_reward + distance_penalty
        print(f"Reward: {total_reward:.3f}")
        return total_reward
    
    def close(self):
        """Clean up resources when the environment is closed"""
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
        except Exception as e:
            print(f"Error closing connections: {e}")
        super().close()

    def render(self):
        """Render the environment (optional - can be implemented for visualization)"""
        pass

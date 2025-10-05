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

class CylinderEnv(gym.Env):
    def __init__(self, size: int = 1000, max_steps: int = 30):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.current_step = 0

        # Connect to Robobo simulator
        self.robobo = Robobo("localhost")
        self.robobo.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()
        
        # RED, GREEN, BLUE, CUSTOM
        self.robobo.setActiveBlobs(True, False, False, False)
        
        # Adjust camera position
        self.robobo.movePanTo(0, 50)
        self.robobo.moveTiltTo(105, 50)
        time.sleep(0.5)
        
        self._object_id = list(self.sim.getObjects())[0] 
        
        # [agent_x, agent_z, target_x, target_z, blob_visible, blob_x, blob_y, blob_size]
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -100.0, -100.0,  # agent position
                -100.0, -100.0,  # target position
                0.0,               # blob_visible (0 o 1)
                0.0, 0.0,          # blob position (0-100)
                0.0                # blob_size (0-100)
            ], dtype=np.float32),
            high=np.array([
                100.0, 100.0,    # agent position
                100.0, 100.0,    # target position
                1.0,               # blob_visible
                100.0, 100.0,      # blob position
                100.0              # blob_size
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Moving up, down, left right
        self.action_space = gym.spaces.Discrete(4)
        self.reset()

    def _get_blob_info(self):
        """Get information about the blob (cylinder)"""
        try:
            blob = self.robobo.readColorBlob(BlobColor.RED)
            agent_x, agent_z = self._get_agent_position()
            target_x, target_z = self._get_object_position()

            if blob is None or blob.size <= 0:
                # Cant see the blob
                return {
                    "visible": 0.0,
                    "x": 50.0,      # Center
                    "y": 50.0,      # Center
                    "size": 0.0
                }
            elif (agent_x - target_x) < 150 and (agent_z - target_z) < 150:
                # We are at the target
                return {
                        "visible": 1.0,
                        "x": 50.0,
                        "y": 50.0,
                        "size": 400 # Max value
                }
            else:
                # Blob detected
                return {
                    "visible": 1.0,
                    "x": float(blob.posx),      # 0-100
                    "y": float(blob.posy),      # 0-100
                    "size": float(blob.size)    # 0-100
                }
        except Exception as e:
            print(f"Error reading blob: {e}")
            return {
                "visible": 0.0,
                "x": 50.0,
                "y": 50.0,
                "size": 0.0
            }


    def _get_agent_position(self):
        """Get current robot position"""
        try:
            # There's always one robot with ID 0, so we only need to catch general exceptions
            agent_location = self.sim.getRobotLocation(0)
            return agent_location["position"]["x"], agent_location["position"]["z"]
        except Exception as e:
            print(f"Error getting agent position: {e}")
            return 0.0, 0.0

    def _get_object_position(self):
        """Get current target object position"""
        try:
            # There's always one object (in other simulations there could be other objects), 
            # but for now we only need to catch general exceptions
            target_location = self.sim.getObjectLocation(self._object_id)
            return target_location["position"]["x"], target_location["position"]["z"]
        except Exception as e:
            print(f"Error getting object position: {e}")
            return 0.0, 0.0

    def _get_obs(self):
        """Convert internal state to observation format with camera data"""
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        blob_info = self._get_blob_info()
        
        return np.array([
            agent_x, agent_z,           
            target_x, target_z,          
            blob_info["visible"],       
            blob_info["x"],             
            blob_info["y"],             
            blob_info["size"]            
        ], dtype=np.float32)

    def _get_info(self):
        """Compute auxiliary information for debugging"""
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        blob_info = self._get_blob_info()
        
        # We use euclidean distance to calculate the distance
        distance = np.sqrt((agent_x - target_x)**2 + (agent_z - target_z)**2)
        
        return {
            "distance": distance,
            "agent_position": (agent_x, agent_z),
            "target_position": (target_x, target_z),
            "step": self.current_step,
            "blob_visible": blob_info["visible"],
            "blob_centered": self._is_blob_centered(blob_info),
            "blob_size": blob_info["size"]
        }

    def _is_blob_centered(self, blob_info, margin=15):
        """Check if blob is centered in camera view"""
        if blob_info["visible"] == 0.0:
            return False
        
        center_x = 50.0
        # Use absolute value to check left and right
        return abs(blob_info["x"] - center_x) < margin
    
   
    
    def _move_target(self):
        """Move the target object"""
        try:
            target_location = self.sim.getObjectLocation(self._object_id)

            target_x = float(target_location["position"]["x"])  # Fuerza float
            target_y = float(target_location["position"]["y"])
            target_z = float(target_location["position"]["z"])
            
            # Movimiento: -100 en x y z (puedes randomizar después)
            new_x = target_x - 40.0
            new_z = target_z - 40.0
            
            # Clipping con float output
            new_x = float(np.clip(new_x, -1000.0, 1000.0))
            new_z = float(np.clip(new_z, -1000.0, 1000.0))
            
            new_position = {
                "x": new_x,
                "y": target_y,
                "z": new_z
            }
            new_rotation = target_location.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
            # Mover
            self.sim.setObjectLocation(self._object_id, new_position, new_rotation)
            self.sim.wait(0.5)  # Wait más largo para sync (ajusta si es lento)
            
        except Exception as e:
            print(f"Error moviendo target: {e}")

    def _calculate_reward(self, current_distance):
        """Calculate reward based on distance and camera observations"""
        blob_info = self._get_blob_info()
        
        # Distance reward
        distance_improvement = self.previous_distance - current_distance
        distance_reward = distance_improvement * 0.1
        
        # Vision reward
        vision_reward = 0.0
        if blob_info["visible"] == 1:
            vision_reward += 0.2
            
            # Reward for how centered is the blob
            center_x = 50.0
            blob_x = blob_info["x"]
            distance_from_center = abs(blob_x - center_x)
            
            # Limit the distance reward to avoid hight values
            centering_reward = 0.3 * (1.0 - distance_from_center / 2.0)
            vision_reward += centering_reward
            
            # Reward for how close is the blob
            size_reward = blob_info["size"] * 0.01
            vision_reward += size_reward
        else:
            # Penalize losing the blob
            vision_reward = -0.3
        
        # Reward for reaching the target
        goal_reward = 0.0
        if current_distance < 150:
            goal_reward = 0.0
        
        # Penalize leaving the target
        distance_penalty = 0.0
        if current_distance > self.previous_distance:
            distance_penalty = -0.2
        
        total_reward = (
            distance_reward + 
            vision_reward + 
            goal_reward + 
            distance_penalty
        )
        
        print(f"Rewards -> Distance: {distance_reward:.2f}, Vision: {vision_reward:.2f}, "
              f"Goal: {goal_reward:.2f}, Penalty: {distance_penalty:.2f}, "
              f"TOTAL: {total_reward:.2f}")
        
        return total_reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode"""
        super().reset(seed=seed)
        
        self.sim.resetSimulation()
        time.sleep(1)
        
        self.robobo.resetColorBlobs()
        
        self.current_step = 0
        self.initial_distance = self._get_info()["distance"]
        self.previous_distance = self.initial_distance

        observation = self._get_obs()
        info = self._get_info()
        
        self.robobo.movePanTo(0, 50)
        self.robobo.moveTiltTo(105, 50)
        time.sleep(0.5)
        

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment"""
        self.current_step += 1

        print(f"\nStep {self.current_step}/{self.max_steps}")

        if self.current_step % 5 == 0:
            print("Moving Target")
            self._move_target()
            # Actualizar la distancia previa después de mover el objetivo
            self.previous_distance = self._get_info()["distance"]


        try:
            if action == 0:  # Move forward
                print("Action: Move forward")
                self.robobo.moveWheelsByTime(20, 20, 0.25)
            elif action == 1:  # Turn left
                print("Action: Turn left")
                self.robobo.moveWheelsByTime(20, -20, 0.25)
            elif action == 2:  # Turn right
                print("Action: Turn right")
                self.robobo.moveWheelsByTime(-20, 20, 0.25)
            elif action == 3:  # Move backward
                print("Action: Move backward")
                self.robobo.moveWheelsByTime(-20, -20, 0.25)

            time.sleep(0.6)
            
        except Exception as e:
            print(f"Error executing action: {e}")
        
        # Get new state
        observation = self._get_obs()
        info = self._get_info()
        current_distance = info["distance"]
        
        print(f"Robot: {info['agent_position']}, Target: {info['target_position']}, "
              f"Distance: {current_distance:.1f}, Blob visible: {info['blob_visible']}, "
              f"Blob size: {info['blob_size']:.1f}")
        
        # Calculate reward
        reward = self._calculate_reward(current_distance)
        
        # Check termination
        terminated = current_distance < 150
        truncated = self.current_step >= self.max_steps
        
        self.previous_distance = current_distance
        
        return observation, reward, terminated, truncated, info

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

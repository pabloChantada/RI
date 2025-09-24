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


from typing import Optional
import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo 
from robobosim.RoboboSim import RoboboSim
import time

class CylinderEnv(gym.Env):
    def __init__(self, size: int = 10, max_steps: int = 30):
        super().__init__()
        
        # The size of the square grid (10x10 by default)
        self.size = size
        self.max_steps = max_steps  # Maximum steps per episode
        self.current_step = 0
        self.acumulated_actions = 0
        self.current_action = None 

        # Connect to Robobo simulator
        self.robobo = Robobo("localhost")
        self.robobo.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()
        
        # Get initial positions (there's always one robot and one object)
        self._object_id = list(self.sim.getObjects())[0] 
        
        # Define observation space - continuous space for robot and target positions
        # Using Box space for continuous x, z coordinates. This indicates the min and max values for each dimension.
        self.observation_space = gym.spaces.Box(
            low=np.array([-10.0, -10.0, -10.0, -10.0], dtype=np.float32),  # [agent_x, agent_z, target_x, target_z]
            high=np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define action space - 4 discrete actions for movement
        self.action_space = gym.spaces.Discrete(4)
        
        # Initialize positions
        self.reset()

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
        """Convert internal state to observation format"""
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        
        return np.array([agent_x, agent_z, target_x, target_z], dtype=np.float32)

    def _get_info(self):
        """Compute auxiliary information for debugging"""
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()

        # Calculate Euclidean distance to target, instead of Manhattan distance
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
        
        # Reset simulation
        self.sim.resetSimulation()
        time.sleep(1) 
        
        # Reset step counter
        self.current_step = 0
        
        # Get initial distance for reward calculation
        self.initial_distance = self._get_info()["distance"]
        self.previous_distance = self.initial_distance
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """Execute one timestep within the environment"""
        self.current_step += 1
              
        match action:
            case 0:
                print("Action: Move forward")
            case 1:
                print("Action: Turn left")
            case 2:
                print("Action: Turn right")
            case 3:
                print("Action: Move backward")

        try:
            if action == 0:  # Move forward
                self.robobo.moveWheelsByTime(20, 20, 0.25)
            elif action == 1:  # Turn left
                self.robobo.moveWheelsByTime(20, -20, 0.25)
            elif action == 2:  # Turn right
                self.robobo.moveWheelsByTime(-20, 20, 0.25)
            elif action == 3:  # Move backward
                self.robobo.moveWheelsByTime(-20, -20, 0.25)

            time.sleep(0.6)
            
        except Exception as e:
            print(f"Error executing action: {e}")

        if self.current_action is not None:
            if action == self.current_action:
                self.acumulated_actions += 1

        self.current_action = action

        # Get new state
        observation = self._get_obs()
        info = self._get_info()
        current_distance = info["distance"]
        print(f"Robot position: {info['agent_position']}, Target position: {info['target_position']}, Distance: {current_distance:.3f}")

        # Calculate reward
        reward = self._calculate_reward(current_distance)
        
        # Check termination conditions
        terminated = current_distance < 150  # Reached target
        truncated = self.current_step >= self.max_steps  # Max steps reached
        
        # Update previous distance for next step
        self.previous_distance = current_distance
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, current_distance):
        """Calculate normalized reward based on distance to target"""
        # Normalize distance change to [-1, 1] range
        max_distance = self.initial_distance if self.initial_distance > 0 else 1.0
        distance_delta = self.previous_distance - current_distance
        normalized_delta = distance_delta / max_distance

        # Reward for getting closer
        distance_reward = normalized_delta

        # Large positive reward for reaching target (normalized)
        if current_distance < 20:
            return 1.0

        if self.acumulated_actions > 3:
            return -50.0
        

        # Penalty for moving away from target
        if current_distance > self.previous_distance:
            distance_penalty = -0.5
        else:
            distance_penalty = 0.0

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
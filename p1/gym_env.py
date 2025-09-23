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

class CylinderEnv(gym.Env):
    def __init__(self, size: int = 10):
        # The size of the square grid (5x5 by default)
        self.size = size

        # Connect to Robobo simulator
        self.robobo = Robobo("localhost")
        self.robobo.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state

        _object_id = list(self.sim.getObjects())[0] # type: ignore
        # ahora puedes usarla
        self._agent_location = self.sim.getRobotLocation(0)
        self._target_location = self.sim.getObjectLocation(_object_id)
        self._target_location_x = self._target_location["position"]["x"]
        self._target_location_z = self._target_location["position"]["z"]
        # Diccionario de diccionarios
        self._init_position_x = self._agent_location["position"]["x"]
        self._init_position_z = self._agent_location["position"]["z"]
    
        self.distance = self._init_position_x - self._target_location_x + self._init_position_z - self._target_location_z

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        # FIX: THIS PART
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        self._action_to_direction = {
            0: self.robobo.moveWheelsByTime(10,10,1),   # Move forward 
            1: self.robobo.moveWheelsByTime(10,-10,1),   # Move right 
            2: self.robobo.moveWheelsByTime(-10,10,1),   # Move left 
            3: self.robobo.moveWheelsByTime(-10,-10,1),  # Move backward
        }

        # FIX: OBSERVATION AND INFO METHODS
        def _get_obs(self):
            """Convert internal state to observation format.

            Returns:
                dict: Observation with agent and target positions
            """
            return {"agent": self._agent_location, "target": self._target_location}

        def _get_info(self):
            """Compute auxiliary information for debugging.

            Returns:
                dict: Info with distance between agent and target
            """
            return {
                "distance": np.linalg.norm(
                    self._agent_location - self._target_location, ord=1
                )
            }

        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            """Start a new episode.

            Args:
                seed: Random seed for reproducible episodes
                options: Additional configuration (unused in this example)

            Returns:
                tuple: (observation, info) for the initial state
            """
            # IMPORTANT: Must call this first to seed the random number generator
            super().reset(seed=seed)

            # Randomly place the agent anywhere on the grid
            self.robobosim.resetSimulation()
            observation = self._get_obs()
            info = self._get_info()

            return observation, info


        def step(self, action):
            """Execute one timestep within the environment.

            Args:
                action: The action to take (0-3 for directions)

            Returns:
                tuple: (observation, reward, terminated, truncated, info)
            """
            # Map the discrete action (0-3) to a movement direction
            direction = self._action_to_direction[action]

            # Update agent position, ensuring it stays within grid bounds
            # np.clip prevents the agent from walking off the edge
            self._agent_location = self.robobosim.getRobotLocation(self.sim.getRobots()[0])
            self._target_location = self.robobosim.getObjectLocation(self.sim.getObjects()[0])
            new_distance = self._agent_location - self._target_location

            # Check if agent reached the target
            terminated = new_distance <= 1

            # We don't use truncation in this simple environment
            # (could add a step limit here if desired)
            truncated = False

            # Simple reward structure: +1 for reaching target, 0 otherwise
            # Alternative: could give small negative rewards for each step to encourage efficiency
            if new_distance < self.distance:
                reward = 1
            else:
                reward = -1

            observation = self._get_obs()
            info = self._get_info()
            self.distance = new_distance

            return observation, reward, terminated, truncated, info
        
        def close(self):
            """Clean up resources when the environment is closed."""
            self.robobo.disconnect()
            self.sim.disconnect()
            super().close()


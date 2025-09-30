from typing import Optional
import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo 
from robobosim.RoboboSim import RoboboSim
import time
import math

class CylinderEnv(gym.Env):
    def __init__(self, size: int = 1000, max_steps: int = 30,
                 randomize_on_reset: bool = True,
                 min_start_dist: float = 250.0,
                 margin: float = 50.0,
                 move_target: bool = False,        # ahora se pasa como parámetro
                 target_speed: float = 40.0,
                 target_mode: str = "random_walk"):
        super().__init__()

        # The size of the square grid (1000x1000 by default)
        self.size = float(size)
        self.half = self.size / 2.0
        self.max_steps = max_steps
        self.current_step = 0

        # Randomization controls
        self.randomize_on_reset = randomize_on_reset
        self.min_start_dist = float(min_start_dist)
        self.margin = float(margin)

        # --- Movimiento del objeto (Objetivo 2) ---
        self.move_target = move_target
        self.target_speed = float(target_speed)
        self.target_mode = target_mode
        self._obj_heading = 0.0
        self._y_object = 0.0
        self._usable_half = self.half - self.margin

        # Connect to Robobo simulator
        self.robobo = Robobo("localhost")
        self.robobo.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()
        
        # Get initial positions (there's always one robot and one object)
        self._object_id = list(self.sim.getObjects())[0] 
        
        # Define observation space - normalizado a [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define action space - 4 discrete actions for movement
        self.action_space = gym.spaces.Discrete(4)
        
        # Initialize positions
        self.reset()

    def _get_agent_position(self):
        try:
            agent_location = self.sim.getRobotLocation(0)
            return agent_location["position"]["x"], agent_location["position"]["z"]
        except Exception as e:
            print(f"Error getting agent position: {e}")
            return 0.0, 0.0

    def _get_object_position(self):
        try:
            target_location = self.sim.getObjectLocation(self._object_id)
            return target_location["position"]["x"], target_location["position"]["z"]
        except Exception as e:
            print(f"Error getting object position: {e}")
            return 0.0, 0.0

    def _normalize(self, x: float, z: float) -> tuple[float, float]:
        """Normaliza coordenadas a [-1, 1] dividiendo por self.half"""
        return float(np.clip(x / self.half, -1.0, 1.0)), float(np.clip(z / self.half, -1.0, 1.0))

    def _get_obs(self):
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        ax, az = self._normalize(agent_x, agent_z)
        tx, tz = self._normalize(target_x, target_z)
        return np.array([ax, az, tx, tz], dtype=np.float32)

    def _get_info(self):
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        distance = float(np.hypot(agent_x - target_x, agent_z - target_z))
        return {
            "distance": distance,
            "agent_position": (agent_x, agent_z),
            "target_position": (target_x, target_z),
            "step": self.current_step
        }

    def _sample_non_colliding_positions(self):
        half = self.size / 2.0 - self.margin
        rx = float(self.np_random.uniform(-half, half))
        rz = float(self.np_random.uniform(-half, half))
        for _ in range(200):
            ox = float(self.np_random.uniform(-half, half))
            oz = float(self.np_random.uniform(-half, half))
            if np.hypot(rx - ox, rz - oz) >= self.min_start_dist:
                return (rx, 0.0, rz), (ox, 0.0, oz)
        ox = np.clip(rx + np.sign(self.np_random.uniform(-1, 1) or 1.0) * self.min_start_dist,
                     -half, half)
        oz = rz
        return (rx, 0.0, rz), (float(ox), 0.0, float(oz))

    def _move_object(self):
        if not self.move_target:
            return
        ox, oz = self._get_object_position()
        if self.target_mode == "random_walk":
            self._obj_heading += float(self.np_random.normal(loc=0.0, scale=np.deg2rad(8.0)))
        elif self.target_mode == "circle":
            self._obj_heading += np.deg2rad(5.0)
        dx = self.target_speed * math.cos(self._obj_heading)
        dz = self.target_speed * math.sin(self._obj_heading)
        nx, nz = float(ox + dx), float(oz + dz)
        bounced = False
        if nx < -self._usable_half or nx > self._usable_half:
            self._obj_heading = np.pi - self._obj_heading
            nx = float(np.clip(nx, -self._usable_half, self._usable_half))
            bounced = True
        if nz < -self._usable_half or nz > self._usable_half:
            self._obj_heading = -self._obj_heading
            nz = float(np.clip(nz, -self._usable_half, self._usable_half))
            bounced = True
        if bounced:
            self._obj_heading += float(self.np_random.normal(0.0, np.deg2rad(5.0)))
        try:
            self.sim.setObjectLocation(self._object_id, nx, self._y_object, nz)
        except Exception as e:
            print(f"Error moving object: {e}")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.sim.resetSimulation()
        time.sleep(0.5)
        try:
            self._object_id = list(self.sim.getObjects())[0]
        except Exception as e:
            print(f"Error getting object id after reset: {e}")
        if self.randomize_on_reset:
            (rx, ry, rz), (ox, oy, oz) = self._sample_non_colliding_positions()
            try:
                self.sim.setRobotLocation(0, rx, ry, rz)
                self.sim.setObjectLocation(self._object_id, ox, oy, oz)
                time.sleep(0.3)
            except Exception as e:
                print(f"Error setting randomized locations: {e}")
        self.current_step = 0
        if self.move_target:
            self._obj_heading = float(self.np_random.uniform(-np.pi, np.pi))
        self.initial_distance = self._get_info()["distance"]
        self.previous_distance = self.initial_distance
        observation = self._get_obs()
        info = self._get_info()
        if info["distance"] < max(150.0, 0.6 * self.min_start_dist):
            (rx, ry, rz), (ox, oy, oz) = self._sample_non_colliding_positions()
            try:
                self.sim.setRobotLocation(0, rx, ry, rz)
                self.sim.setObjectLocation(self._object_id, ox, oy, oz)
                time.sleep(0.3)
                observation = self._get_obs()
                info = self._get_info()
                self.initial_distance = info["distance"]
                self.previous_distance = self.initial_distance
            except Exception as e:
                print(f"Error fixing too-close start: {e}")
        return observation, info

    def step(self, action):
        self.current_step += 1
        try:
            if action == 0:
                self.robobo.moveWheelsByTime(20, 20, 0.5)
            elif action == 1:
                self.robobo.moveWheelsByTime(20, -20, 0.5)
            elif action == 2:
                self.robobo.moveWheelsByTime(-20, 20, 0.5)
            elif action == 3:
                self.robobo.moveWheelsByTime(-20, -20, 0.5)
            time.sleep(0.6)
        except Exception as e:
            print(f"Error executing action: {e}")
        self._move_object()
        time.sleep(0.05)
        observation = self._get_obs()
        info = self._get_info()
        current_distance = info["distance"]
        reward = self._calculate_reward(current_distance)
        terminated = current_distance < 150
        truncated = self.current_step >= self.max_steps
        self.previous_distance = current_distance
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, current_distance):
        distance_reward = (self.initial_distance - current_distance) * 10.0
        if current_distance < 10.0:
            return 1.0
        distance_penalty = -0.5 if current_distance > self.previous_distance else 0.0
        return float(distance_reward + distance_penalty)

    def close(self):
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
        except Exception as e:
            print(f"Error closing connections: {e}")
        super().close()

    def render(self):
        pass

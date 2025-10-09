"""
Custom Gymnasium Environment for Red Cylinder Search Task

This environment simulates a robot (Robobo) searching for a red cylinder in a square arena.
The robot uses camera blob detection to locate and approach the target.

Environment Specifications:
    - Arena size: 2000x2000 units ([-1000, 1000] in both X and Z axes)
    - Observation space: 8-dimensional normalized vector
    - Action space: Continuous wheel velocities [-1, 1] for each wheel
    - Episode termination: Goal reached or max steps exceeded
"""

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
    Square environment for red cylinder search using Gymnasium structure.

    Observation Space (8 dimensions, all normalized):
        - Agent position (x, z): [-1, 1]
        - Target position (x, z): [-1, 1]
        - Blob visible: {0, 1}
        - Blob position (x, y): [-1, 1] (centered at camera center)
        - Blob size: [0, 1]
    
    Action Space (2 dimensions):
        - Continuous wheel velocities: [-1, 1] for left and right wheels
    """
    
    def __init__(self, size: int = 1000, max_steps: int = 30):
        """
        Initialize the custom environment.
        
        Args:
            size: Maximum camera blob size threshold
            max_steps: Maximum steps per episode, defaults to 30
        """
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Goal reached threshold 
        self.goal_threshold = 150.0
        
        # Target movement frequency (every N steps)
        self.target_move_frequency = 5 

        # Connect to Robobo simulator
        self.robobo = Robobo("localhost")
        self.robobo.connect()
        self.sim = RoboboSim("localhost")
        self.sim.connect()
        
        # Configure blob detection (RED only)
        self.robobo.setActiveBlobs(True, False, False, False)
        
        # Setup camera position
        self._setup_camera()
        
        # Get target object ID (cylinder)
        self._object_id = list(self.sim.getObjects())[0]
        
        # OBSERVATION SPACE: All normalized values
        # [agent_x, agent_z, target_x, target_z, 
        #  blob_visible, blob_x, blob_y, blob_size]
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -1.0, -1.0,  # agent position
                -1.0, -1.0,  # target position
                0.0,         # blob_visible (0 or 1)
                -1.0, -1.0,  # blob position (centered at 0)
                0.0          # blob_size
            ], dtype=np.float32),
            high=np.array([
                1.0, 1.0,    # agent position
                1.0, 1.0,    # target position
                1.0,         # blob_visible (0 or 1)
                1.0, 1.0,    # blob position (centered at 0)
                1.0          # blob_size
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # ACTION SPACE: Continuous wheel velocities
        # [left_wheel_velocity, right_wheel_velocity] in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Action conversion parameters
        self.speed = 20.0        # Scale to convert [-1,1] to motor speed
        self.wheels_time = 0.25  # Time duration for wheel actions
        
        # Tracking variables
        self.previous_distance = None
        self.initial_distance = None

    def _setup_camera(self):
        """Configure robot camera position (tilt down to see floor)."""
        self.robobo.movePanTo(0, 50)      # Reset pan
        self.robobo.moveTiltTo(105, 50)   # Tilt to max downward position
        time.sleep(0.2)

    def _get_blob_info(self):
        """
        Get information about detected blob from camera.
        
        Returns:
            dict: Blob information with keys 'visible', 'x', 'y', 'size'
        """
        try:
            blob = self.robobo.readColorBlob(BlobColor.RED)
            agent_x, agent_z = self._get_agent_position()
            target_x, target_z = self._get_object_position()
            
            distance_to_target = np.sqrt((agent_x - target_x)**2 + (agent_z - target_z)**2)

            # Case 1: No blob detected
            if blob is None or blob.size <= 0:
                return {
                    "visible": 0.0,
                    "x": 50.0,      # Image center
                    "y": 50.0,
                    "size": 0.0
                }
            
            # Case 2: Very close to target (goal reached)
            elif distance_to_target < self.goal_threshold:
                return {
                    "visible": 1.0,
                    "x": 50.0,      # Centered
                    "y": 50.0,
                    "size": 400.0   # Maximum size
                }
            
            # Case 3: Blob detected normally
            else:
                return {
                    "visible": 1.0,
                    "x": float(blob.posx),      # Range [0-100]
                    "y": float(blob.posy),      # Range [0-100]
                    "size": float(blob.size)    # Range [0-100+]
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
        """
        Get current robot position.
        
        Returns:
            tuple: (x, z) coordinates
        """
        try:
            agent_location = self.sim.getRobotLocation(0)
            # We dont need to track rotation and "y" axis in this case
            return agent_location["position"]["x"], agent_location["position"]["z"]
        except Exception as e:
            print(f"Error getting agent position: {e}")
            return 0.0, 0.0

    def _get_object_position(self):
        """
        Get current target object position.
        
        Returns:
            tuple: (x, z) coordinates
        """
        try:
            target_location = self.sim.getObjectLocation(self._object_id)
            # We dont need to track rotation and "y" axis in this case
            return target_location["position"]["x"], target_location["position"]["z"]
        except Exception as e:
            print(f"Error getting object position: {e}")
            return 0.0, 0.0

    def _get_obs(self):
        """
        Generate current normalized observation.
        
        Returns:
            np.ndarray: Normalized observation vector (8 dimensions)
        """
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        blob_info = self._get_blob_info()
        
        # Normalize positions: [-1000, 1000] -> [-1, 1]
        agent_x_norm = np.clip(agent_x / 1000.0, -1.0, 1.0)
        agent_z_norm = np.clip(agent_z / 1000.0, -1.0, 1.0)
        target_x_norm = np.clip(target_x / 1000.0, -1.0, 1.0)
        target_z_norm = np.clip(target_z / 1000.0, -1.0, 1.0)
        
        # Normalize blob position centered at camera center: [0, 100] -> [-1, 1]
        blob_x_norm = np.clip((blob_info["x"] - 50.0) / 50.0, -1.0, 1.0)
        blob_y_norm = np.clip((blob_info["y"] - 50.0) / 50.0, -1.0, 1.0)
        
        # Normalize blob size: [0, 400] -> [0, 1]
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
        Generate auxiliary debugging information.
        
        Returns:
            dict: Current state information
        """
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        blob_info = self._get_blob_info()
        
        # Euclidean distance
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
        Check if blob is centered in camera view.
        
        Args:
            blob_info: Blob information dictionary
            margin: Tolerance margin in pixels
            
        Returns:
            bool: True if blob is centered within margin
        """
        if blob_info["visible"] == 0.0:
            return False
        
        center_x = 50.0
        return abs(blob_info["x"] - center_x) < margin
    
    def _move_target(self):
        """
        Move target in a curved trajectory.
        
        The cylinder moves in both X (lateral) and Z (forward/backward) directions
        simultaneously. Direction changes occur when arena boundaries are reached.
        """
        try:
            target_location = self.sim.getObjectLocation(self._object_id)
            
            target_x = float(target_location["position"]["x"])
            target_y = float(target_location["position"]["y"])
            target_z = float(target_location["position"]["z"])
            
            # Initialize curved movement parameters
            if not hasattr(self, 'target_direction'):
                # Choose curve direction: -1 (left curve) or +1 (right curve)
                self.target_direction = random.choice([-1, 1])
                
                # Always move backwards (negative Z)
                self.z_direction = -1

                # Set movement speeds
                self.target_speed_x = random.uniform(4.0, 6.0)  # Lateral speed
                self.target_speed_z = random.uniform(3.0, 5.0)  # Forward/backward speed

                direction_name = "RIGHT" if self.target_direction > 0 else "LEFT"
                print(f"\n  [Target initialized - Curved backward and to {direction_name}]")
                print(f"      Speed X: {self.target_speed_x:.1f}, Speed Z: {self.target_speed_z:.1f}")

            # Calculate simultaneous curved movement
            move_x = self.target_direction * self.target_speed_x  # Lateral
            move_z = -abs(self.target_speed_z)                    # Always backward (negative)

            # Calculate new position
            new_x = target_x + move_x
            new_z = target_z + move_z
            
            # Boundary detection and direction change
            MARGIN_X = 150.0  # Lateral margin
            MARGIN_Z = 150.0  # Front/back margin
            
            # Bounce on lateral boundaries (X)
            if new_x <= -1000.0 + MARGIN_X:
                new_x = -1000.0 + MARGIN_X
                self.target_direction = 1  # Now curve right
                print(f"\n  [LEFT LATERAL bounce - Now curves RIGHT]")
            elif new_x >= 1000.0 - MARGIN_X:
                new_x = 1000.0 - MARGIN_X
                self.target_direction = -1  # Now curve left
                print(f"\n  [RIGHT LATERAL bounce - Now curves LEFT]")
            
            # Bounce on front/back boundaries (Z)
            if new_z <= -1000.0 + MARGIN_Z:
                new_z = -1000.0 + MARGIN_Z
                self.z_direction = 1  # Now moves away
                print(f"\n  [FRONT bounce - Now moves AWAY]")
            elif new_z >= 1000.0 - MARGIN_Z:
                new_z = 1000.0 - MARGIN_Z
                self.z_direction = -1  # Now moves toward robot
                print(f"\n  [BACK bounce - Now moves TOWARD ROBOT]")
            
            # Ensure values are within range
            new_x = float(np.clip(new_x, -1000.0, 1000.0))
            new_z = float(np.clip(new_z, -1000.0, 1000.0))
            
            new_position = {
                "x": new_x,
                "y": target_y,
                "z": new_z
            }
            
            # The rotation is needed for the function but isnt really used
            new_rotation = target_location.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
            
            # Update object position
            self.sim.setObjectLocation(self._object_id, new_position, new_rotation)
            
        except Exception as e:
            print(f"Error moving target: {e}")

    def _calculate_reward(self, current_distance, blob_info):
        """
        Calculate reward based on distance and vision.
        
        Reward strategy:
        1. Reward for approaching target
        2. Bonus for keeping blob visible and centered
        3. Penalty for losing sight of blob
        4. Large reward for reaching goal
        
        Args:
            current_distance: Current distance to target
            blob_info: Detected blob information
        
        Returns:
            float: Total reward
        """
        reward = 0.0
        
        # 1. Reward for distance reduction
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - current_distance
            reward += distance_improvement * 0.05
        
        # 2. Reward for blob visibility
        if blob_info["visible"] == 1.0:
            reward += 0.2
            
            # Bonus for centering blob
            center_x = 50.0
            blob_x = blob_info["x"]
            distance_from_center = abs(blob_x - center_x)
            centering_reward = 0.2 * (1.0 - min(distance_from_center / 50.0, 1.0))
            reward += centering_reward
            
            # Bonus for blob size (closer = larger)
            size_reward = min(blob_info["size"] / 400.0, 1.0) * 0.1
            reward += size_reward
        else:
            # Penalty for losing blob
            reward -= 0.15
        
        # 3. Reward for reaching goal
        if current_distance < self.goal_threshold:
            reward += 10.0
        
        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment for a new episode.
        
        Args:
            seed: Seed for reproducibility
            options: Additional options
        
        Returns:
            observation: Initial observation
            info: Initial information
        """
        super().reset(seed=seed)
        
        # Reset simulation
        self.sim.resetSimulation()
        time.sleep(0.5)
        
        # Reset blob detection
        self.robobo.resetColorBlobs()
        
        # Reset camera
        self._setup_camera()
        
        # Reset counters
        self.current_step = 0
        info = self._get_info()
        self.initial_distance = info["distance"]
        self.previous_distance = self.initial_distance
        
        observation = self._get_obs()
        
        return observation, info

    def step(self, action):
        """
        Execute an action in the environment.
        
        Args:
            action: Array of 2 elements [left_wheel_vel, right_wheel_vel] in [-1, 1]
        
        Returns:
            observation: New observation
            reward: Obtained reward
            terminated: Whether episode ended (goal reached)
            truncated: Whether episode was truncated (max_steps reached)
            info: Additional information
        """
        self.current_step += 1

        # Move target every step 
        # Only if target_move_frequency is low (for phase 2)
        if hasattr(self, 'target_move_frequency') and self.target_move_frequency < 100:
            self._move_target()
            # Update previous distance after moving target
            # to avoid penalizing agent for target movement
            info_temp = self._get_info()
            self.previous_distance = info_temp["distance"]

        # Validate and process action
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.size != 2:
            raise ValueError(f"Action must have 2 elements, received {action.size}")

        # Clip action values to valid range
        left_vel = float(np.clip(action[0], -1.0, 1.0))
        right_vel = float(np.clip(action[1], -1.0, 1.0))

        # Execute action in simulator
        try:
            left_motor = int(np.clip(left_vel * self.speed, -100, 100))
            right_motor = int(np.clip(right_vel * self.speed, -100, 100))
            
            self.robobo.moveWheelsByTime(right_motor, left_motor, self.wheels_time)
            # time.sleep(0.25)
            
        except Exception as e:
            print(f"Error executing action: {e}")

        # Get new state
        observation = self._get_obs()
        info = self._get_info()
        current_distance = info["distance"]
        blob_info = self._get_blob_info()
        
        # Calculate reward
        reward = self._calculate_reward(current_distance, blob_info)
        
        # Check termination conditions
        terminated = current_distance < self.goal_threshold  # Goal reached
        truncated = self.current_step >= self.max_steps      # Time limit
        
        # Logging
        print(f"Step {self.current_step}/{self.max_steps} | "
              f"Action: L={left_vel:.2f} R={right_vel:.2f} | "
              f"Dist: {current_distance:.1f} | "
              f"Blob: {'1' if blob_info['visible'] else '0'} | "
              f"Reward: {reward:.2f}")
        
        # Update previous distance
        self.previous_distance = current_distance
        
        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources when closing environment."""
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
            print("Connections closed successfully")
        except Exception as e:
            print(f"Error closing connections: {e}")
        super().close()

    def render(self):
        """Render environment (not implemented)."""
        pass
from typing import Optional
import gymnasium as gym
import numpy as np
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import time
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR


class CustomEnv(gym.Env):
    """
    Square environment for red cylinder search using Gymnasium structure.

    Observation space (12 dims):
        - Agent position (x, z): [-1, 1]
        - Target position (x, z): [-1, 1]
        - Blob visible: {0, 1}
        - Blob position (x, y): [-1, 1]
        - Blob size: [0, 1]
        - IR sensors (4): [0, 1] (FrontC, FrontL, FrontR, FrontRR)

    Action space (2 dims): Continuous wheel velocities [-1, 1]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        size: int = 1000,
        max_steps: int = 30,
        sim_host: str = "localhost",
        sim_speed: float = 4.0,
        verbose: bool = False,
    ):
        """
        Args:
            size: camera blob size threshold concept (kept for compatibility)
            max_steps: max steps per episode
            sim_host: host for robobo/robobosim
            sim_speed: multiplier for simulator speed (e.g. 4.0)
            verbose: if True prints debug logs
        """
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.current_step = 0
        self.verbose = verbose

        # Goal reached threshold
        self.goal_threshold = 150.0

        # Collision avoidance threshold
        self.collision_threshold = 0.3  # Normalized IR value

        self.robobo = Robobo(sim_host)
        self.robobo.connect()
        self.sim = RoboboSim(sim_host)
        self.sim.connect()

        # Default start position
        self.default_start = {"x": -1000.0, "y": 39.0, "z": -400.0}
        self.sim.setRobotLocation(0, self.default_start)

        # Configure blob detection (RED only)
        self.robobo.setActiveBlobs(True, False, False, False)

        # Setup camera
        self._setup_camera()

        # Get target object id
        try:
            objs = list(self.sim.getObjects())
            self._object_id = objs[0]
        except Exception:
            self._object_id = None

        # OBSERVATION SPACE: 12 dimensions (8 vision + 4 IR sensors)
        # [agent_x, agent_z, target_x, target_z,
        #  blob_visible, blob_x, blob_y, blob_size,
        #  ir_front_c, ir_front_l, ir_front_r, ir_front_rr]
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [-1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Action conversion
        self.speed = 20.0
        self.wheels_time = 0.25

        # Tracking
        self.previous_distance = None
        self.initial_distance = None

    def _setup_camera(self):
        """Tilt camera downward so robot sees floor."""
        try:
            self.robobo.movePanTo(0, 50)
            self.robobo.moveTiltTo(105, 50)
            time.sleep(0.2)
        except Exception:
            pass

    def _get_ir_sensors(self):
        """
        Read and normalize IR sensor values for collision avoidance.

        IMPORTANT: Robobo IR sensors work inversely:
        - Low raw value = close obstacle (DANGER)
        - High raw value = far away or no obstacle (SAFE)

        We normalize to:
        - 0.0 = safe (far/no obstacle)
        - 1.0 = danger (very close obstacle)

        Returns:
            dict: Normalized IR sensor readings [0, 1]
                  0 = no obstacle, 1 = very close obstacle
        """
        try:
            # Read IR sensors (range typically 0-1000+)
            # Higher values = further away
            ir_front_c = self.robobo.readIRSensor(IR.FrontC)
            ir_front_l = self.robobo.readIRSensor(IR.FrontL)
            ir_front_r = self.robobo.readIRSensor(IR.FrontR)
            ir_front_rr = self.robobo.readIRSensor(IR.FrontRR)

            if self.verbose:
                print(
                    f"[IR RAW] C={ir_front_c:.1f}, L={ir_front_l:.1f}, "
                    f"R={ir_front_r:.1f}, RR={ir_front_rr:.1f}"
                )

            # Define thresholds
            # If IR > safe_distance, consider it safe (normalize to 0)
            # If IR < danger_distance, consider it dangerous (normalize to 1)
            safe_distance = 100.0  # Above this = safe (no obstacle)
            danger_distance = 10.0  # Below this = very dangerous

            def normalize_ir(raw_value):
                """
                Normalize IR sensor reading inversely.
                High raw value (far) -> 0.0 (safe)
                Low raw value (close) -> 1.0 (danger)
                """
                if raw_value >= safe_distance:
                    return 0.0  # Safe, no obstacle
                elif raw_value <= danger_distance:
                    return 1.0  # Danger, very close
                else:
                    # Linear interpolation between danger and safe
                    # Invert: as raw increases, normalized decreases
                    normalized = 1.0 - (raw_value - danger_distance) / (
                        safe_distance - danger_distance
                    )
                    return float(np.clip(normalized, 0.0, 1.0))

            return {
                "front_c": normalize_ir(ir_front_c),
                "front_l": normalize_ir(ir_front_l),
                "front_r": normalize_ir(ir_front_r),
                "front_rr": normalize_ir(ir_front_rr),
            }
        except Exception as e:
            if self.verbose:
                print(f"Error reading IR sensors: {e}")
            return {
                "front_c": 0.0,
                "front_l": 0.0,
                "front_r": 0.0,
                "front_rr": 0.0,
            }

    def _get_blob_info(self):
        """Return dict: visible (0/1), x,y (0-100), size (>=0). Handles exceptions."""
        try:
            blob = self.robobo.readColorBlob(BlobColor.RED)
        except Exception:
            blob = None

        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        distance_to_target = (
            np.sqrt((agent_x - target_x) ** 2 + (agent_z - target_z) ** 2)
            if (agent_x is not None and target_x is not None)
            else 9999.0
        )

        if blob is None or getattr(blob, "size", 0) <= 0:
            return {"visible": 0.0, "x": 50.0, "y": 50.0, "size": 0.0}
        # If very close treat as centered big blob
        if distance_to_target < self.goal_threshold:
            return {"visible": 1.0, "x": 50.0, "y": 50.0, "size": 400.0}
        return {
            "visible": 1.0,
            "x": float(blob.posx),
            "y": float(blob.posy),
            "size": float(blob.size),
        }

    def _get_agent_position(self):
        """Return (x,z) or (0,0) on error."""
        try:
            loc = self.sim.getRobotLocation(0)
            return loc["position"]["x"], loc["position"]["z"]
        except Exception:
            return 0.0, 0.0

    def _get_object_position(self):
        """Return (x,z) for the target object or (0,0) on error."""
        try:
            if self._object_id is None:
                objs = list(self.sim.getObjects())
                if objs:
                    self._object_id = objs[0]
                else:
                    return 0.0, 0.0
            loc = self.sim.getObjectLocation(self._object_id)
            return loc["position"]["x"], loc["position"]["z"]
        except Exception:
            return 0.0, 0.0

    def _get_obs(self):
        """
        Generate current normalized observation with IR sensors.

        Returns:
            np.ndarray: Normalized observation vector (12 dimensions)
        """
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        blob_info = self._get_blob_info()
        ir_sensors = self._get_ir_sensors()

        agent_x_norm = np.clip(agent_x / 1000.0, -1.0, 1.0)
        agent_z_norm = np.clip(agent_z / 1000.0, -1.0, 1.0)
        target_x_norm = np.clip(target_x / 1000.0, -1.0, 1.0)
        target_z_norm = np.clip(target_z / 1000.0, -1.0, 1.0)
        blob_x_norm = np.clip((blob_info["x"] - 50.0) / 50.0, -1.0, 1.0)
        blob_y_norm = np.clip((blob_info["y"] - 50.0) / 50.0, -1.0, 1.0)
        blob_size_norm = np.clip(blob_info["size"] / 400.0, 0.0, 1.0)

        return np.array(
            [
                agent_x_norm,
                agent_z_norm,
                target_x_norm,
                target_z_norm,
                blob_info["visible"],
                blob_x_norm,
                blob_y_norm,
                blob_size_norm,
                ir_sensors["front_c"],
                ir_sensors["front_l"],
                ir_sensors["front_r"],
                ir_sensors["front_rr"],
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        agent_x, agent_z = self._get_agent_position()
        target_x, target_z = self._get_object_position()
        blob_info = self._get_blob_info()
        ir_sensors = self._get_ir_sensors()
        distance = np.sqrt((agent_x - target_x) ** 2 + (agent_z - target_z) ** 2)

        return {
            "distance": distance,
            "agent_position": (agent_x, agent_z),
            "target_position": (target_x, target_z),
            "step": self.current_step,
            "blob_visible": bool(blob_info["visible"]),
            "blob_centered": self._is_blob_centered(blob_info),
            "blob_size": blob_info["size"],
            "ir_sensors": ir_sensors,
            "near_obstacle": max(ir_sensors.values()) > self.collision_threshold,
        }

    def _is_blob_centered(self, blob_info, margin=15):
        if blob_info["visible"] == 0.0:
            return False
        return abs(blob_info["x"] - 50.0) < margin

    def _calculate_reward(self, current_distance, blob_info, ir_sensors):
        """
        Calculate reward based on distance, vision, and collision avoidance.

        Args:
            current_distance: Current distance to target
            blob_info: Detected blob information
            ir_sensors: IR sensor readings

        Returns:
            float: Total reward
        """
        reward = 0.0

        # 1. Reward for distance improvement
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - current_distance
            reward += distance_improvement * 0.05

        # 2. Reward for blob visibility
        if blob_info["visible"] == 1.0:
            reward += 0.2
            distance_from_center = abs(blob_info["x"] - 50.0)
            centering_reward = 0.2 * (1.0 - min(distance_from_center / 50.0, 1.0))
            reward += centering_reward
            size_reward = min(blob_info["size"] / 400.0, 1.0) * 0.1
            reward += size_reward
        else:
            reward -= 0.15

        # 3. Penalty for being too close to obstacles (collision avoidance)
        max_ir = max(ir_sensors.values())
        if max_ir > self.collision_threshold:
            # Penalty proportional to proximity
            collision_penalty = (max_ir - self.collision_threshold) * 0.5
            reward -= collision_penalty

            if self.verbose:
                print(
                    f"  [WARNING] Near obstacle! Max IR: {max_ir:.2f}, Penalty: -{collision_penalty:.2f}"
                )

        # 4. Large reward for reaching goal
        if current_distance < self.goal_threshold:
            reward += 10.0

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        try:
            self.sim.resetSimulation()
        except Exception:
            pass
        time.sleep(0.3)
        try:
            self.robobo.resetColorBlobs()
        except Exception:
            pass
        self._setup_camera()
        # Change robot starting position
        self.sim.setRobotLocation(0, self.default_start)
        self.current_step = 0
        info = self._get_info()
        self.initial_distance = info["distance"]
        self.previous_distance = self.initial_distance
        obs = self._get_obs()
        if self.verbose:
            print(f"[ENV] reset. initial_distance={self.initial_distance:.1f}")
        return obs, info

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.size != 2:
            raise ValueError("Action must have 2 elements [left, right]")
        left_vel = float(np.clip(action[0], -1.0, 1.0))
        right_vel = float(np.clip(action[1], -1.0, 1.0))
        try:
            left_motor = int(np.clip(left_vel * self.speed, -100, 100))
            right_motor = int(np.clip(right_vel * self.speed, -100, 100))
            self.robobo.moveWheelsByTime(left_motor, right_motor, self.wheels_time)
        except Exception:
            # swallow motor errors during parallel evaluation
            pass

        observation = self._get_obs()
        info = self._get_info()
        current_distance = info["distance"]
        blob_info = self._get_blob_info()
        ir_sensors = self._get_ir_sensors()

        reward = self._calculate_reward(current_distance, blob_info, ir_sensors)
        terminated = current_distance < self.goal_threshold
        truncated = self.current_step >= self.max_steps

        if self.verbose and (self.current_step % 10 == 0 or terminated):
            obstacle_warning = " [NEAR OBSTACLE!]" if info["near_obstacle"] else ""
            print(
                f"Step {self.current_step}/{self.max_steps} | L={left_vel:.2f} R={right_vel:.2f} | "
                f"Dist={current_distance:.1f} | Blob={int(blob_info['visible'])} | "
                f"IR_max={max(ir_sensors.values()):.2f} | R={reward:.2f}{obstacle_warning}"
            )
        self.previous_distance = current_distance
        return observation, reward, terminated, truncated, info

    def close(self):
        try:
            self.robobo.disconnect()
            self.sim.disconnect()
        except Exception:
            pass
        super().close()

    def render(self, mode="human"):
        # No additional rendering beyond simulator
        return

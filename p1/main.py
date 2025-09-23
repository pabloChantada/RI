import gymnasium as gym
from cylinder_env import CylinderEnv  

# Personalized gymnasium environment
env = CylinderEnv()
observation, info = env.reset()

print(f"Starting observation: {observation}")
print(f"Initial info: {info}")

episode_over = False
total_reward = 0
step_count = 0
max_steps = 20 

print("\nStarting episode...")
while not episode_over and step_count < max_steps:
    action = env.action_space.sample()  # Random action
    print(f"Step {step_count + 1}: Taking action {action}")
    
    # Execute action
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1
    
    print(f"  Reward: {reward:.3f}, Distance: {info['distance']:.3f}")
    
    episode_over = terminated or truncated

print(f"Total steps: {step_count}")
print(f"Total reward: {total_reward:.3f}")
print(f"Final distance: {info['distance']:.3f}")

# Clean up
env.close()
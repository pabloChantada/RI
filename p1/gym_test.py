import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
print("Action space:", env.action_space)
print("Sampled action:", env.action_space.sample())


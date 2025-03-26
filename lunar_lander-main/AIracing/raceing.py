import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

# Load the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="human")

# Load the trained model (Make sure to replace 'car_racing_model.zip' with the actual model file)
model_path = "car_racing_model.zip"
model = PPO.load(model_path)

# Run the model in the environment
done = False
obs, _ = env.reset()

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

# Close the environment
env.close()

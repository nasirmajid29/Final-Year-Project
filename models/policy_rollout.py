import gym
import rlbench.gym

import torch
from simple_pointnet_pp import Net, SAModule, GlobalSAModule

model = Net()

model = torch.load("reach_red_ball_pointnet++.pt")
model.eval()

env = gym.make('reach_target-state-v0', render_mode='human')
 
training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    action = model(obs) 
    obs, reward, terminate, _ = env.step(action) # env.action_space.sample())
    env.render()  # Note: rendering increases step time.

print('Done')
env.close()
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from stable_baselines3.common.evaluation import evaluate_policy

from IPython import display
import matplotlib.pyplot as plt

import time

env = gym.make("CartPole-v1", render_mode='rgb_array')

model = PPO(MlpPolicy, env, verbose=0)
#-------------------------------

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)

print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
#-------------------------------

model.learn(total_timesteps=10_000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#-------------------------------

obs, info = env.reset()
epoch = 10000

data = {
    'images': [],
    'states': [],
    'actions': []
}

i = 0
terminated = False
while not terminated and i < epoch:
    action, _ = model.predict(obs)

    screen = env.render()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    data['images'].append(screen)
    data['states'].append(obs)
    data['actions'].append(action)

    obs, _, terminated, _, _ = env.step(action)
    i += 1

print(i)


i_step = 5
for i, image in enumerate(data['images']):
    if i % i_step == 0:
        plt.imshow(image)
        plt.show()
        print(data['states'][i])
        print(data['actions'][i])
        time.sleep(0.5)


plt.close()
env.close()

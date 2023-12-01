import gymnasium as gym
import numpy as np

from data_generator import generate_data

from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.ppo.policies import MlpPolicy

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise


def CartPole():

    env = gym.make("CartPole-v1", render_mode='rgb_array')

    model = PPO(MlpPolicy, env, verbose=1)
    # -------------------------------

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)

    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    # -------------------------------

    model.learn(total_timesteps=int(1.0e5), log_interval=5)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    generate_data('CartPole', env, model)

    env.close()


if __name__ == '__main__':
    CartPole()
    #MountainCar()
    #Pendulum()
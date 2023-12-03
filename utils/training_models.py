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


def MountainCar():
    env = gym.make("MountainCar-v0", render_mode="rgb_array")

    dqn_model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        train_freq=16,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        target_update_interval=600,
        learning_starts=1000,
        buffer_size=10000,
        batch_size=128,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=2,
    )

    mean_reward, std_reward = evaluate_policy(
        dqn_model,
        dqn_model.get_env(),
        deterministic=True,
        n_eval_episodes=20,
    )

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    dqn_model.learn(int(1.2e5), log_interval=10)

    mean_reward, std_reward = evaluate_policy(dqn_model, dqn_model.get_env(), deterministic=True, n_eval_episodes=20)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    generate_data('MountainCar', env, dqn_model)

    env.close()


def Pendulum():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    vec_env = model.get_env()

    generate_data('Pendulum', vec_env, model, vec_env=True)

    env.close()


if __name__ == '__main__':
    #CartPole()
    #MountainCar()
    Pendulum()

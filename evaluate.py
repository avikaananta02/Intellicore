import numpy as np
import torch

from config import Config
from env import make_env
from agent import DoubleDQNAgent


def evaluate(model_path, n_episodes=10, render=False, record=False):

    config = Config()

    env, preprocessor, stacker = make_env(
        config.ENV_NAME,
        config.FRAME_SIZE,
        config.FRAME_STACK
    )

    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DoubleDQNAgent(
        n_frames=config.FRAME_STACK,
        n_actions=n_actions,
        config=config,
        device=device,
    )

    agent.load(model_path)

    rewards = []

    for episode in range(n_episodes):

        obs, info = env.reset()

        state = stacker.reset(
            preprocessor.process(obs)
        )

        done = False
        episode_reward = 0

        while not done:

            if render:
                env.render()

            with torch.no_grad():
                state_tensor = torch.FloatTensor(
                    state
                ).unsqueeze(0).to(device)

                action = (
                    agent.online_net(state_tensor)
                    .argmax(dim=1)
                    .item()
                )

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            state = stacker.append(
                preprocessor.process(next_obs)
            )

            episode_reward += reward

        rewards.append(episode_reward)

        print(
            f"Episode {episode + 1}: Reward = {episode_reward}"
        )

    env.close()

    return rewards

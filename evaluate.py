"""
Evaluation script.
Loads a trained model and runs episodes.
"""

import numpy as np
import torch

from config import Config
from env import make_env
from agent import Agent


def evaluate(model_path, n_episodes=10, render=False):

    config = Config()

    env, preprocessor, stacker = make_env(
        config.ENV_NAME,
        config.FRAME_SIZE,
        config.FRAME_STACK
    )

    if render:
        env.close()

        env, preprocessor, stacker = make_env(
            config.ENV_NAME,
            config.FRAME_SIZE,
            config.FRAME_STACK
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    agent = Agent(
        n_frames=config.FRAME_STACK,
        n_actions=env.action_space.n,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon=0.0,  # Greedy evaluation
        device=device
    )

    # Load trained weights
    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    agent.online_net.load_state_dict(
        checkpoint
    )

    agent.target_net.load_state_dict(
        checkpoint
    )

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

            action = agent.select_action(state)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            next_state = stacker.append(
                preprocessor.process(next_obs)
            )

            episode_reward += reward

            state = next_state

        rewards.append(episode_reward)

        print(
            f"Episode {episode + 1} | "
            f"Reward: {episode_reward}"
        )

    env.close()

    print("\nEvaluation Complete")
    print(f"Mean Reward : {np.mean(rewards):.2f}")
    print(f"Std Reward  : {np.std(rewards):.2f}")

    return rewards


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10
    )

    parser.add_argument(
        "--render",
        action="store_true"
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render
    )

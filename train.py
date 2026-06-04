"""
Main training loop.
"""

import torch

from config import Config
from env import make_env
from env import clip_reward
from agent import Agent


def train():

    config = Config()

    # ----------------------------------
    # Environment
    # ----------------------------------
    env, preprocessor, stacker = make_env(
        config.ENV_NAME,
        config.FRAME_SIZE,
        config.FRAME_STACK
    )

    n_actions = env.action_space.n

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ----------------------------------
    # Agent
    # ----------------------------------
    agent = Agent(
        n_frames=config.FRAME_STACK,
        n_actions=n_actions,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon=config.EPS_START,
        device=device
    )

    # ----------------------------------
    # Metrics
    # ----------------------------------
    episode_rewards = []
    losses = []

    episode_reward = 0.0
    episode_count = 0

    # ----------------------------------
    # Initial state
    # ----------------------------------
    obs, info = env.reset()

    state = stacker.reset(
        preprocessor.process(obs)
    )

    # ----------------------------------
    # Training Loop
    # ----------------------------------
    for step in range(
        1,
        config.TOTAL_STEPS + 1
    ):

        # Select action
        action = agent.select_action(state)

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        reward = clip_reward(reward)

        next_state = stacker.append(
            preprocessor.process(next_obs)
        )

        # Store transition
        if hasattr(agent, "store_transition"):
            agent.store_transition(
                state,
                action,
                reward,
                next_state,
                done
            )

        # Learn
        loss = None

        if hasattr(agent, "learn"):
            loss = agent.learn()

        if loss is not None:
            losses.append(loss)

        episode_reward += reward

        state = next_state

        # --------------------------
        # Target network update
        # --------------------------
        if step % config.TARGET_UPDATE == 0:

            if hasattr(agent, "update_target_network"):
                agent.update_target_network()

        # --------------------------
        # Save checkpoint
        # --------------------------
        if (
            step % config.SAVE_EVERY == 0
            and hasattr(agent, "save")
        ):
            agent.save(
                f"checkpoint_{step}.pth"
            )

        # --------------------------
        # Logging
        # --------------------------
        if step % config.LOG_EVERY == 0:

            avg_loss = (
                sum(losses[-100:]) /
                max(1, len(losses[-100:]))
                if len(losses) > 0
                else 0
            )

            print(
                f"Step {step} | "
                f"Episodes {episode_count} | "
                f"Recent Avg Loss {avg_loss:.4f}"
            )

        # --------------------------
        # Episode end
        # --------------------------
        if done:

            episode_rewards.append(
                episode_reward
            )

            episode_count += 1

            print(
                f"Episode {episode_count} | "
                f"Reward = {episode_reward:.2f}"
            )

            obs, info = env.reset()

            state = stacker.reset(
                preprocessor.process(obs)
            )

            episode_reward = 0.0

    env.close()

    return episode_rewards


if __name__ == "__main__":
    rewards = train()

    print(
        f"Training finished. "
        f"Episodes completed: {len(rewards)}"
    )

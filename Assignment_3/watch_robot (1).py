import sys
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ContinualBench"))
from continual_bench.envs import ContinualBenchEnv
from ppo_env_wrapper import ContinualBenchGymnasiumWrapper
from ppo_1 import ActorCritic

# ── Config ────────────────────────────────────────────────────────────────────
TASK         = "faucet"   # change to: button, door, window, peg, block
NUM_EPISODES = 5
MAX_STEPS    = 500
MODEL_PATH = "ppo_1_faucet.pth"      # set to "ppo_1_faucet.pth" if you saved weights
DEVICE       = "cpu"
# ─────────────────────────────────────────────────────────────────────────────

def make_env(task=TASK):
    base_env = ContinualBenchEnv(render_mode="rgb_array", seed=42)
    base_env.set_task(task)
    env = ContinualBenchGymnasiumWrapper(base_env, task)
    return env

def load_agent(env, model_path=None):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    network = ActorCritic(obs_dim, act_dim).to(DEVICE)

    if model_path and os.path.exists(model_path):
        network.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded weights from {model_path}")
    else:
        print("No saved weights — using random (untrained) policy.")

    network.eval()
    return network

def watch(network, env, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    fig.tight_layout()

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        img_display = None

        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, _, _, _ = network.get_action_and_value(obs_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward

            # Render frame and display via matplotlib
            frame = env.render()
            if frame is not None:
                if img_display is None:
                    img_display = ax.imshow(frame)
                else:
                    img_display.set_data(frame)
                ax.set_title(f"Episode {ep} | Step {step+1} | Reward: {total_reward:.2f}")
                plt.pause(0.02)

            if terminated or truncated:
                break

        print(f"Episode {ep} | Steps: {step+1} | Total Reward: {total_reward:.3f}")

    plt.ioff()
    plt.show()
    env.close()

if __name__ == "__main__":
    print(f"Watching agent on task: {TASK}")
    env     = make_env(task=TASK)
    network = load_agent(env, model_path=MODEL_PATH)
    watch(network, env)

import sys
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ContinualBench"))
from continual_bench.envs import ContinualBenchEnv
from ppo_env_wrapper import ContinualBenchGymnasiumWrapper
from ppo_1 import ActorCritic

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cpu"

# Each phase shows: which weights to load + which task to run
# This lets you SEE forgetting visually
PHASES = [
    # (title,                    weights_file,              task_to_watch)
    ("After training FAUCET",   "continual_faucet.pth",    "faucet"),   # should be good
    ("After training BUTTON",   "continual_button.pth",    "faucet"),   # faucet forgotten!
    ("After training BUTTON",   "continual_button.pth",    "button"),   # button learned
    ("After training DOOR",     "continual_door.pth",      "faucet"),   # faucet more forgotten
    ("After training DOOR",     "continual_door.pth",      "door"),     # door learned
]

EPISODES_PER_PHASE = 1
MAX_STEPS          = 300
# ─────────────────────────────────────────────────────────────────────────────

def make_env(task):
    base_env = ContinualBenchEnv(render_mode="rgb_array", seed=42)
    base_env.set_task(task)
    return ContinualBenchGymnasiumWrapper(base_env, task)

def load_network(env, weights_file):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    network = ActorCritic(obs_dim, act_dim).to(DEVICE)

    if os.path.exists(weights_file):
        network.load_state_dict(torch.load(weights_file, map_location=DEVICE))
        print(f"Loaded: {weights_file}", flush=True)
    else:
        print(f"NOT FOUND: {weights_file} — using random policy", flush=True)

    network.eval()
    return network

def watch_phase(title, weights_file, task):
    print(f"\n{'='*50}", flush=True)
    print(f"  {title}", flush=True)
    print(f"  Task: {task}", flush=True)
    print(f"{'='*50}", flush=True)

    env = make_env(task)
    network = load_network(env, weights_file)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    fig.tight_layout()
    img_display = None

    for ep in range(1, EPISODES_PER_PHASE + 1):
        obs, _ = env.reset()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, _, _ = network.get_action_and_value(obs_tensor)

            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += reward

            frame = env.render()
            if frame is not None:
                if img_display is None:
                    img_display = ax.imshow(frame)
                else:
                    img_display.set_data(frame)
                ax.set_title(
                    f"{title}\nTask: {task} | Step {step+1} | Reward: {total_reward:.1f}",
                    fontsize=10
                )
                plt.pause(0.02)

            if terminated or truncated:
                break

        print(f"  Episode {ep} | Steps: {step+1} | Total Reward: {total_reward:.1f}", flush=True)

    plt.ioff()
    plt.close()
    env.close()

if __name__ == "__main__":
    print("Watching Continual Learning — Catastrophic Forgetting Demo", flush=True)
    print("You will see the robot perform each task after each training phase.", flush=True)
    print("Notice how faucet performance DROPS after training on button/door!\n", flush=True)

    for title, weights_file, task in PHASES:
        watch_phase(title, weights_file, task)
        input(f"\nPress ENTER to continue to next phase...")

    print("\nDemo complete! Did you see the forgetting? 🤖", flush=True)

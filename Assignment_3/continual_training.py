"""
continual_training.py

Demonstrates catastrophic forgetting by training one PPO network
sequentially on multiple tasks and evaluating performance on all tasks
after each training phase.

Flow:
  1. Train on Task 1 (faucet)  → evaluate all tasks
  2. Train on Task 2 (button)  → evaluate all tasks  ← faucet reward drops = forgetting!
  3. Train on Task 3 (door)    → evaluate all tasks
  ...
"""

import sys
import os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ContinualBench"))
from continual_bench.envs import ContinualBenchEnv
from ppo_env_wrapper import ContinualBenchGymnasiumWrapper
from ppo_1 import ActorCritic, PPOAgent, compute_gae

# ── Config ─────────────────────────────────────────────────────────────────
TASKS          = ["faucet", "button", "door"]   # add more: "window","peg","block"
TRAIN_STEPS    = 100000   # steps per task (increase for better learning)
NUM_STEPS      = 2048
BATCH_SIZE     = 64
N_EPOCHS       = 10
EVAL_EPISODES  = 3        # episodes to evaluate each task
EVAL_STEPS     = 300      # max steps per eval episode
DEVICE         = "cpu"
# ───────────────────────────────────────────────────────────────────────────

def make_env(task):
    base_env = ContinualBenchEnv(render_mode="rgb_array", seed=42)
    base_env.set_task(task)
    return ContinualBenchGymnasiumWrapper(base_env, task)

def evaluate(network, task, episodes=EVAL_EPISODES, max_steps=EVAL_STEPS):
    """Evaluate the network on a task and return average reward."""
    env = make_env(task)
    total_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, _, _ = network.get_action_and_value(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(total_reward)

    env.close()
    return np.mean(total_rewards)

def train_on_task(agent, task, total_timesteps=TRAIN_STEPS):
    """Train the agent on a single task."""
    env = make_env(task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    num_envs = 1
    num_updates = total_timesteps // NUM_STEPS

    obs_buf     = torch.zeros(NUM_STEPS, num_envs, obs_dim).to(DEVICE)
    actions_buf = torch.zeros(NUM_STEPS, num_envs, act_dim).to(DEVICE)
    logprobs_buf = torch.zeros(NUM_STEPS, num_envs).to(DEVICE)
    rewards_buf  = torch.zeros(NUM_STEPS, num_envs).to(DEVICE)
    dones_buf    = torch.zeros(NUM_STEPS, num_envs).to(DEVICE)
    values_buf   = torch.zeros(NUM_STEPS, num_envs).to(DEVICE)

    next_obs, _ = env.reset()
    next_obs  = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    next_done = torch.zeros(num_envs).to(DEVICE)

    for update in range(1, num_updates + 1):
        for step in range(NUM_STEPS):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.network.get_action_and_value(next_obs)

            actions_buf[step]  = action
            logprobs_buf[step] = logprob
            values_buf[step]   = value.view(-1)

            next_obs_np, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            done = terminated or truncated
            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            next_done = torch.tensor([float(done)]).to(DEVICE)

            if done:
                next_obs_np, _ = env.reset()
                next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                next_done = torch.zeros(num_envs).to(DEVICE)

        with torch.no_grad():
            next_value = agent.network.get_value(next_obs).view(-1)

        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf,
            next_value, next_done, agent.gamma, agent.gae_lambda
        )

        b_obs        = obs_buf.view(-1, obs_dim)
        b_actions    = actions_buf.view(-1, act_dim)
        b_logprobs   = logprobs_buf.view(-1)
        b_advantages = advantages.view(-1)
        b_returns    = returns.view(-1)
        b_values     = values_buf.view(-1)

        total_samples = NUM_STEPS * num_envs
        for _ in range(N_EPOCHS):
            indices = torch.randperm(total_samples)
            for start in range(0, total_samples, BATCH_SIZE):
                mb_idx = indices[start: start + BATCH_SIZE]
                agent.update(
                    b_obs[mb_idx], b_actions[mb_idx], b_logprobs[mb_idx],
                    b_advantages[mb_idx], b_returns[mb_idx], b_values[mb_idx],
                )

        if update % 10 == 0 or update == num_updates:
            print(f"  [{task}] Update {update}/{num_updates}", flush=True)

    env.close()

def print_results_table(results):
    """Print a nice table of results."""
    tasks = list(results[list(results.keys())[0]].keys())
    header = f"{'After Training':<20}" + "".join(f"{t:>12}" for t in tasks)
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))
    for phase, rewards in results.items():
        row = f"{phase:<20}" + "".join(f"{rewards[t]:>12.1f}" for t in tasks)
        print(row)
    print("="*len(header))

if __name__ == "__main__":
    print("="*60, flush=True)
    print("  Continual Learning — Catastrophic Forgetting Demo", flush=True)
    print("="*60, flush=True)

    # Get obs/act dims from first task
    tmp_env = make_env(TASKS[0])
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    tmp_env.close()
    print(f"obs_dim={obs_dim}, act_dim={act_dim}", flush=True)

    # One shared agent for all tasks
    agent = PPOAgent(obs_dim, act_dim, device=DEVICE)

    results = {}  # results[phase][task] = avg_reward

    # Evaluate before any training (random policy baseline)
    print("\n[Baseline] Evaluating random policy...", flush=True)
    baseline = {task: evaluate(agent.network, task) for task in TASKS}
    results["Baseline (random)"] = baseline
    for t, r in baseline.items():
        print(f"  {t}: {r:.1f}", flush=True)

    # Sequential training
    for i, train_task in enumerate(TASKS):
        print(f"\n{'='*60}", flush=True)
        print(f"Phase {i+1}: Training on '{train_task}'...", flush=True)
        print(f"{'='*60}", flush=True)

        train_on_task(agent, train_task, total_timesteps=TRAIN_STEPS)

        # Save weights after each task
        save_path = f"continual_{train_task}.pth"
        torch.save(agent.network.state_dict(), save_path)
        print(f"Saved: {save_path}", flush=True)

        # Evaluate ALL tasks after training this task
        print(f"\n[Eval after {train_task}] Evaluating all tasks...", flush=True)
        phase_results = {}
        for eval_task in TASKS:
            avg_r = evaluate(agent.network, eval_task)
            phase_results[eval_task] = avg_r
            print(f"  {eval_task}: {avg_r:.1f}", flush=True)

        results[f"After {train_task}"] = phase_results

    # Final results table
    print_results_table(results)
    print("\nDone! Check the table above to see catastrophic forgetting.", flush=True)
    print("Tasks trained earlier will show LOWER rewards after training on new tasks.", flush=True)

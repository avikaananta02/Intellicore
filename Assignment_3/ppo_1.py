import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import sys
import os

# Ensure ContinualBench can be imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ContinualBench"))
from continual_bench.envs import ContinualBenchEnv
from ppo_env_wrapper import ContinualBenchGymnasiumWrapper

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action spaces.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action).sum(1),
            dist.entropy().sum(1),
            self.critic(x),
        )

def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            non_terminal = 1.0 - next_done.float()
            next_val = next_value
        else:
            non_terminal = 1.0 - dones[t + 1].float()
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * non_terminal * last_gae_lam

    returns = advantages + values
    return advantages, returns

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.network = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def update(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, clip_vloss=True):
        _, new_logprobs, entropy, new_values = self.network.get_action_and_value(b_obs, b_actions)

        ratio = torch.exp(new_logprobs - b_logprobs)

        # Normalize advantages
        adv = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        new_values = new_values.view(-1)
        if clip_vloss:
            v_clipped = b_values + torch.clamp(new_values - b_values, -self.clip_coef, self.clip_coef)
            v_loss_unclipped = (new_values - b_returns) ** 2
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        total_loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "total_loss": total_loss.item(),
        }

def train_ppo_example(env, total_timesteps=10000, num_steps=2048, batch_size=64, n_epochs=10, device="cpu"):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    num_envs = 1  # single env

    agent = PPOAgent(obs_dim, act_dim, device=device)

    # Rollout storage
    obs_buf    = torch.zeros(num_steps, num_envs, obs_dim).to(device)
    actions_buf = torch.zeros(num_steps, num_envs, act_dim).to(device)
    logprobs_buf = torch.zeros(num_steps, num_envs).to(device)
    rewards_buf  = torch.zeros(num_steps, num_envs).to(device)
    dones_buf    = torch.zeros(num_steps, num_envs).to(device)
    values_buf   = torch.zeros(num_steps, num_envs).to(device)

    # Initial reset
    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)  # (1, obs_dim)
    next_done = torch.zeros(num_envs).to(device)

    num_updates = total_timesteps // num_steps

    for update in range(1, num_updates + 1):
        # ── Collect rollout ───────────────────────────────────────────────
        for step in range(num_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.network.get_action_and_value(next_obs)
            
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.view(-1)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            rewards_buf[step] = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).unsqueeze(0).to(device)
            next_done = torch.tensor([float(done)]).to(device)

            if done:
                next_obs_np, _ = env.reset()
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32).unsqueeze(0).to(device)
                next_done = torch.zeros(num_envs).to(device)

        # ── Compute advantages & returns ──────────────────────────────────
        with torch.no_grad():
            next_value = agent.network.get_value(next_obs).view(-1)

        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf,
            next_value, next_done,
            agent.gamma, agent.gae_lambda
        )

        # ── Flatten batch tensors ─────────────────────────────────────────
        b_obs       = obs_buf.view(-1, obs_dim)
        b_actions   = actions_buf.view(-1, act_dim)
        b_logprobs  = logprobs_buf.view(-1)
        b_advantages = advantages.view(-1)
        b_returns   = returns.view(-1)
        b_values    = values_buf.view(-1)

        # ── Optimise ──────────────────────────────────────────────────────
        total_samples = num_steps * num_envs
        all_metrics = []

        for _ in range(n_epochs):
            indices = torch.randperm(total_samples)
            for start in range(0, total_samples, batch_size):
                mb_idx = indices[start: start + batch_size]
                metrics = agent.update(
                    b_obs[mb_idx], b_actions[mb_idx], b_logprobs[mb_idx],
                    b_advantages[mb_idx], b_returns[mb_idx], b_values[mb_idx],
                )
                all_metrics.append(metrics)

        # ── Logging ───────────────────────────────────────────────────────
        avg_pg   = np.mean([m["pg_loss"]    for m in all_metrics])
        avg_vf   = np.mean([m["v_loss"]     for m in all_metrics])
        avg_ent  = np.mean([m["entropy"]    for m in all_metrics])
        avg_tot  = np.mean([m["total_loss"] for m in all_metrics])
        print(
            f"Update {update}/{num_updates} | "
            f"pg_loss={avg_pg:.4f}  v_loss={avg_vf:.4f}  "
            f"entropy={avg_ent:.4f}  total={avg_tot:.4f}",
            flush=True
        )

    # Save trained weights
    import os
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_1_faucet.pth")
    torch.save(agent.network.state_dict(), save_path)
    print(f"Weights saved to {save_path}", flush=True)

    return agent

if __name__ == "__main__":
    print("Testing custom PPO on ContinualBench 'faucet' task...", flush=True)

    # 1. Initialize ContinualBenchEnv with render_mode='rgb_array' and a seed.
    print("Loading environment...", flush=True)
    base_env = ContinualBenchEnv(render_mode="rgb_array", seed=42)

    # 2. Set the task to 'faucet'
    base_env.set_task("faucet")
    print("Environment loaded!", flush=True)

    # 3. Wrap the env in ContinualBenchGymnasiumWrapper
    env = ContinualBenchGymnasiumWrapper(base_env, "faucet")

    print("Starting Training...", flush=True)
    agent = train_ppo_example(env, total_timesteps=500000, num_steps=2048, batch_size=64, n_epochs=10)
    print("Training complete!", flush=True)

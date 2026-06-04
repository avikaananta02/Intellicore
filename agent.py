import random
import numpy as np
import torch
import torch.nn.functional as F

from network import QNetwork
from replay_buffer import PrioritizedReplayBuffer


class Agent:

    def __init__(
        self,
        n_frames,
        n_actions,
        lr,
        gamma,
        epsilon,
        device,
        buffer_size=100000,
        batch_size=32
    ):

        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.batch_size = batch_size

        self.online_net = QNetwork(
            n_frames,
            n_actions
        ).to(device)

        self.target_net = QNetwork(
            n_frames,
            n_actions
        ).to(device)

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=lr
        )

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size
        )

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state = (
            torch.FloatTensor(state)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.online_net(state)

        return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.buffer.store(
            state,
            action,
            reward,
            next_state,
            done
        )

    def learn(self):

        if len(self.buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            is_weights
        ) = self.buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(
            states
        ).to(self.device)

        actions = torch.LongTensor(
            actions
        ).unsqueeze(1).to(self.device)

        rewards = torch.FloatTensor(
            rewards
        ).to(self.device)

        next_states = torch.FloatTensor(
            next_states
        ).to(self.device)

        dones = torch.FloatTensor(
            dones
        ).to(self.device)

        is_weights = torch.FloatTensor(
            is_weights
        ).to(self.device)

        current_q = (
            self.online_net(states)
            .gather(1, actions)
            .squeeze(1)
        )

        with torch.no_grad():

            next_actions = (
                self.online_net(next_states)
                .argmax(dim=1, keepdim=True)
            )

            next_q = (
                self.target_net(next_states)
                .gather(1, next_actions)
                .squeeze(1)
            )

            target_q = rewards + (
                self.gamma
                * next_q
                * (1 - dones)
            )

        td_errors = (
            target_q - current_q
        ).detach().cpu().numpy()

        loss = (
            is_weights
            * F.mse_loss(
                current_q,
                target_q,
                reduction="none"
            )
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.update_priorities(
            indices,
            np.abs(td_errors)
        )

        return loss.item()

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

    def save(self, path):

        torch.save(
            self.online_net.state_dict(),
            path
        )

    def load(self, path):

        self.online_net.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

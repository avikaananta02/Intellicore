import random
import torch
from network import QNetwork


class Agent:
    def __init__(self, n_frames, n_actions, lr, gamma, epsilon, device):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

        self.online_net = QNetwork(n_frames, n_actions).to(device)
        self.target_net = QNetwork(n_frames, n_actions).to(device)

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=lr
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

    def learn(self):
        raise NotImplementedError(
            "Will be completed after replay_buffer.py is available."
        )

    def update_target_network(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

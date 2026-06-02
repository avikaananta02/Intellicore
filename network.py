"""
Neural network architecture for the DQN agent.
A standard CNN that takes stacked frames and outputs
Q-values for each action.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Convolutional Q-Network.

    Input:  (batch, n_frames, 84, 84)
    Output: (batch, n_actions)
    """

    def __init__(self, n_frames, n_actions):
        """
        Define the layers:
          - 3 convolutional layers (with ReLU)
          - 2 fully connected layers
        Args:
            n_frames: int, number of stacked frames (input channels)
            n_actions: int, number of possible actions
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: torch.Tensor of shape (batch, n_frames, 84, 84)
        Returns:
            torch.Tensor of shape (batch, n_actions)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

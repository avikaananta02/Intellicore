"""
Prioritized Experience Replay buffer using a Sum Tree.
"""

import numpy as np


class SumTree:
    """
    Binary tree where each leaf stores a priority and
    internal nodes store the sum of their children.
    Allows O(log n) sampling proportional to priority.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write_index = 0
        self.size = 0

    def total(self):
        """Return the total sum of all priorities."""
        return self.tree[0]

    def add(self, priority, data):
        """
        Store a transition with the given priority.
        """
        tree_index = self.write_index + self.capacity - 1

        self.data[self.write_index] = data
        self.update(tree_index, priority)

        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0

        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_index, priority):
        """
        Update the priority of a leaf node and propagate.
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, value):
        """
        Retrieve a leaf by sampling a value in [0, total()).
        Returns:
            (tree_index, priority, data)
        """
        parent = 0

        while True:
            left = 2 * parent + 1
            right = left + 1

            if left >= len(self.tree):
                leaf = parent
                break

            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right

        data_index = leaf - self.capacity + 1

        return (
            leaf,
            self.tree[leaf],
            self.data[data_index]
        )


class PrioritizedReplayBuffer:
    """
    Replay buffer that samples transitions proportional
    to their TD-error priority.
    """

    def __init__(
        self,
        capacity,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_steps=200_000,
        epsilon=1e-6,
    ):
        self.tree = SumTree(capacity)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon

        self.max_priority = 1.0
        self.step_count = 0

    def _get_beta(self):
        """Linearly anneal beta from beta_start to beta_end."""

        beta = self.beta_start + (
            (self.beta_end - self.beta_start)
            * min(1.0, self.step_count / self.beta_steps)
        )

        return beta

    def store(self, state, action, reward, next_state, done):
        """
        Store a transition with max priority.
        """

        transition = (
            state,
            action,
            reward,
            next_state,
            done,
        )

        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        """
        Sample a batch proportional to priorities.
        """

        self.step_count += 1

        beta = self._get_beta()

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        indices = []
        priorities = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)

            value = np.random.uniform(start, end)

            idx, priority, data = self.tree.get(value)

            state, action, reward, next_state, done = data

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            priorities.append(priority)
            indices.append(idx)

        probabilities = np.array(priorities) / self.tree.total()

        weights = (self.tree.size * probabilities) ** (-beta)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        """
        Update priorities after learning.
        """

        td_errors = np.abs(td_errors)

        for idx, error in zip(indices, td_errors):
            priority = (error + self.epsilon) ** self.alpha

            self.tree.update(idx, priority)

            self.max_priority = max(
                self.max_priority,
                priority
            )

    def __len__(self):
        return self.tree.size

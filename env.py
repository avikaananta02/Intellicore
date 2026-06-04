"""
Environment wrappers for Atari preprocessing.
Handles frame grayscale conversion, resizing,
stacking, and reward clipping.
"""

import gymnasium as gym
import ale_py
import numpy as np
import cv2

gym.register_envs(ale_py)


class FramePreprocessor:
    """Converts a raw RGB frame to grayscale and resizes to (size, size)."""

    def __init__(self, size=84):
        self.size = size

    def process(self, frame):
        """
        Args:
            frame: np.ndarray of shape (210, 160, 3)

        Returns:
            np.ndarray of shape (size, size),
            dtype float32,
            values in [0, 1]
        """

        # Convert RGB -> Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize to target size
        resized = cv2.resize(
            gray,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA
        )

        # Normalize to [0,1]
        processed = resized.astype(np.float32) / 255.0

        return processed


class FrameStacker:
    """Maintains a stack of the last n preprocessed frames."""

    def __init__(self, n_frames=4, frame_size=84):
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.frames = None

    def reset(self, initial_frame):
        """
        Initialise stack by repeating first frame.

        Args:
            initial_frame:
                np.ndarray of shape (frame_size, frame_size)

        Returns:
            np.ndarray of shape
            (n_frames, frame_size, frame_size)
        """

        self.frames = np.stack(
            [initial_frame] * self.n_frames,
            axis=0
        )

        return self.frames

    def append(self, frame):
        """
        Push a new frame and remove the oldest.

        Args:
            frame:
                np.ndarray of shape
                (frame_size, frame_size)

        Returns:
            np.ndarray of shape
            (n_frames, frame_size, frame_size)
        """

        self.frames = np.concatenate(
            (
                self.frames[1:],
                np.expand_dims(frame, axis=0)
            ),
            axis=0
        )

        return self.frames


def clip_reward(reward):
    """
    Reward clipping.

    Positive reward -> +1
    Negative reward -> -1
    Zero reward     -> 0
    """

    return np.sign(reward)


def make_env(env_name, frame_size=84, n_frames=4):
    """
    Create Atari environment along with
    preprocessor and frame stacker.

    Args:
        env_name: str
        frame_size: int
        n_frames: int

    Returns:
        env, preprocessor, stacker
    """

    env = gym.make(env_name)

    preprocessor = FramePreprocessor(size=frame_size)

    stacker = FrameStacker(
        n_frames=n_frames,
        frame_size=frame_size
    )

    return env, preprocessor, stacker


# ---------------------------------------------------
# Quick Test
# ---------------------------------------------------

if __name__ == "__main__":

    env, preprocessor, stacker = make_env(
        "ALE/Boxing-v5",
        frame_size=84,
        n_frames=4
    )

    obs, info = env.reset()

    frame = preprocessor.process(obs)

    state = stacker.reset(frame)

    print("Processed frame shape:", frame.shape)
    print("Stacked state shape:", state.shape)

    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, info = env.step(action)

    next_frame = preprocessor.process(next_obs)

    next_state = stacker.append(next_frame)

    clipped_reward = clip_reward(reward)

    print("Next state shape:", next_state.shape)
    print("Original reward:", reward)
    print("Clipped reward:", clipped_reward)

    env.close()

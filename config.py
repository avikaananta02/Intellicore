"""
Centralized hyperparameter configuration.
All project constants live here.
"""


class Config:

    # ==================================================
    # Environment
    # ==================================================
    ENV_NAME = "ALE/Boxing-v5"

    FRAME_SIZE = 84
    FRAME_STACK = 4

    # ==================================================
    # Neural Network / DQN
    # ==================================================
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    BATCH_SIZE = 32

    # ==================================================
    # Epsilon-Greedy Exploration
    # ==================================================
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY_STEPS = 100_000

    # ==================================================
    # Prioritized Experience Replay
    # ==================================================
    BUFFER_SIZE = 100_000

    PER_ALPHA = 0.6

    PER_BETA_START = 0.4
    PER_BETA_END = 1.0
    PER_BETA_STEPS = 200_000

    PER_EPSILON = 1e-6

    # ==================================================
    # Training
    # ==================================================
    TRAIN_START = 10_000

    TOTAL_STEPS = 500_000

    TARGET_UPDATE = 1_000

    SAVE_EVERY = 50_000

    LOG_EVERY = 1_000

    # ==================================================
    # Evaluation
    # ==================================================
    EVAL_EPISODES = 10

    # ==================================================
    # Checkpoints
    # ==================================================
    MODEL_PATH = "boxing_dqn.pth"

    # ==================================================
    # Reproducibility
    # ==================================================
    SEED = 42

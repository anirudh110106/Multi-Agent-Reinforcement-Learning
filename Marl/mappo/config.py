"""
config.py

Configuration file for MAPPO training on CAGE Challenge 4 (CC4).
Modify hyperparameters here instead of changing them throughout the code.
"""

# ==========================================================
# Environment
# ==========================================================

NUM_AGENTS = 5

EPISODE_LENGTH = 100          # Must match EnterpriseScenarioGenerator(steps=100)

MISSION_PHASES = 3


# ==========================================================
# Observation / Action Dimensions
# ==========================================================

# Blue Agents 0-3
SMALL_OBS_DIM = 92
SMALL_ACTION_DIM = 82

# Blue Agent 4
LARGE_OBS_DIM = 210
LARGE_ACTION_DIM = 242

# Shared-policy dimensions
#
# We pad every observation to 210 features
# and every action distribution to 242 actions.
#
# Invalid actions are masked before sampling.
#
OBS_DIM = LARGE_OBS_DIM
ACTION_DIM = LARGE_ACTION_DIM


# ==========================================================
# MAPPO Hyperparameters
# ==========================================================
TOTAL_EPISODES = 1000                             


ROLLOUT_STEPS = 512
UPDATE_EPOCHS = 5 #(was 10)
MINIBATCH_SIZE = 256
LEARNING_RATE = 3e-4

#( separate learning rates)
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 5e-5


GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5


# ==========================================================
# Neural Network
# ==========================================================

HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 5 # (change back to 2 later!!!!)
ACTIVATION = "relu"
DEVICE = "cuda"

# ==========================================================
# Logging
# ==========================================================

PRINT_EVERY = 10
SAVE_EVERY = 500

CHECKPOINT_DIR = "checkpoints/aam_test1"
LOG_DIR = "evaluation/aam_test1"


# ==========================================================
# Randomness
# ==========================================================

SEED = 42
# PPO Training
UPDATE_EPOCHS = 5
MINIBATCH_SIZE = 256
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
PPO_CLIP = 0.2
MAX_GRAD_NORM = 0.5
# Add these
VALUE_CLIP = True #(was false)          # Optional value clipping
NORMALIZE_ADVANTAGES = True # Standard MAPPO practice



# ==========================================================
# Curriculum Learning
# ==========================================================
USE_VALUE_NORM = True
CURRICULUM_ENABLED = True

CURRICULUM_STAGES = [
    (0, "RandomSelectRedAgent"),
    (4000, "FiniteStateRedAgent"),
]
CURRICULUM_SWITCH_EPISODE = 200


CURRICULUM_SCHEDULE = [
    (100, 0.10),   # 80% Random, 20% Finite
    (300, 0.40),   # 60% Random, 40% Finite
    (700, 0.80),   # 40% Random, 60% Finite
    (1000, 1.00),   # 100% Finite
    (10000, 1.00),  # 100% Finite
]




# attention :-
EMBED_DIM = 256
NUM_HEADS = 4


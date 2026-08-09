"""
evaluate.py

Evaluation for the current CC4 MAPPO implementation.

Compatible with:

    - GNN + Attention actor
    - V2 structured communication
    - 7-field communication
    - Dynamic Trust
    - padded observations
    - action masking
    - centralized critic
    - ValueNorm
    - checkpointed trust state
    - one-step communication delay

Evaluation is completely frozen:

    NO gradients
    NO PPO updates
    NO rollout buffer
    NO trust updates
    NO MessageEvaluator
    NO ground-truth feedback to the policy

The evaluation path mirrors the training-time rollout:

    observation
        +
    messages from t-1
        +
    trust
        |
        v
      MAPPO
        |
        +---- action
        |
        +---- message for t+1
        |
        v
      CybORG


Usage
-----

Single checkpoint, both Red agents:

    python -m Marl.mappo.evaluate \
        --checkpoint checkpoints/attention_test_curriculum/mappo_final.pt \
        --episodes 100

FiniteStateRedAgent only:

    python -m Marl.mappo.evaluate \
        --checkpoint checkpoints/attention_test_curriculum/mappo_final.pt \
        --episodes 100 \
        --red-agent finite

RandomSelectRedAgent only:

    python -m Marl.mappo.evaluate \
        --checkpoint checkpoints/attention_test_curriculum/mappo_final.pt \
        --episodes 100 \
        --red-agent random

Checkpoint sweep:

    python -m Marl.mappo.evaluate \
        --sweep checkpoints/attention_test_curriculum \
        --episodes 50

By default evaluation is greedy.

Use:

    --stochastic

to sample actions from the policy instead.
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.distributions import Categorical

from .env import CC4Env
from .mappo import MAPPO

from .train import (
    pad_observation,
    build_action_mask,
    episode_is_done,
)

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    ACTION_DIM,
    EPISODE_LENGTH,
)

from CybORG.Agents import (
    RandomSelectRedAgent,
    FiniteStateRedAgent,
)


# ==========================================================
# Red agents
# ==========================================================

RED_AGENTS = {
    "random": RandomSelectRedAgent,
    "finite": FiniteStateRedAgent,
}


# ==========================================================
# Communication state helper
# ==========================================================

def get_current_communication_for_agent(
    ppo,
    receiver_id,
    previous_messages,
):
    """
    Return exactly the communication state available to one
    receiver before it chooses its action.

    This mirrors train.py.

    previous_messages are generated at timestep t-1,
    therefore they are the only messages available at timestep t.

    Returns
    -------

    received_messages:
        [NUM_AGENTS, COMMUNICATION_DIM]

    trust_weights:
        [NUM_AGENTS]

    or:

        None, None

    at the first timestep.
    """

    if previous_messages is None:
        return None, None

    received_messages = (
        ppo.get_messages_for_agent(
            receiver_id=receiver_id,
            messages=previous_messages,
        )
    )

    trust_weights = (
        ppo.get_trust_for_agent(
            receiver_id=receiver_id,
        )
    )

    return (
        received_messages,
        trust_weights,
    )


# ==========================================================
# Greedy action
# ==========================================================

@torch.no_grad()
def select_action_greedy(
    ppo,
    observation,
    action_mask,
    received_messages,
    trust_weights,
):
    """
    Greedy evaluation.

    Uses the same actor path as training:

        observation
             +
        received communication
             +
        trust
             |
             v
          actor
             |
             v
        action mask
             |
             v
        argmax action
    """

    observation = ppo._to_tensor(
        observation,
        dtype=torch.float32,
    )

    action_mask = ppo._to_tensor(
        action_mask,
        dtype=torch.bool,
    )

    logits = ppo.actor_forward(
        observations=observation,
        action_masks=action_mask,
        received_messages=received_messages,
        trust_weights=trust_weights,
    )

    action = torch.argmax(
        logits,
        dim=-1,
    )

    return int(action.item())


# ==========================================================
# Stochastic action
# ==========================================================

@torch.no_grad()
def select_action_stochastic(
    ppo,
    observation,
    action_mask,
    global_state,
    agent_id,
    received_messages,
    trust_weights,
):
    """
    Stochastic evaluation.

    This uses MAPPO's normal select_action() path.
    """

    action, _, _, _ = ppo.select_action(
        observation=observation,
        action_mask=action_mask,
        global_state=global_state,
        agent_id=agent_id,
        received_messages=received_messages,
        trust_weights=trust_weights,
    )

    return int(action)


# ==========================================================
# Run episodes
# ==========================================================

def run_episodes(
    ppo,
    red_agent_class,
    num_episodes,
    deterministic=True,
    base_seed=100_000,
):
    """
    Evaluate a frozen MAPPO policy against one Red agent.

    Important:

        Trust is NOT updated.

        Ground truth is NOT queried.

        The checkpoint's trust matrix remains frozen.

    Communication timing exactly matches training:

        timestep t:
            receive message generated at t-1
            choose action
            generate message

        timestep t+1:
            receive that message
    """

    # ------------------------------------------------------
    # Create initial environment
    # ------------------------------------------------------

    env = CC4Env(
        red_agent_class=red_agent_class,
    )

    agent_names = sorted(
        env.possible_agents
    )

    if len(agent_names) != NUM_AGENTS:
        raise RuntimeError(
            "Unexpected number of Blue agents: "
            f"expected {NUM_AGENTS}, "
            f"got {len(agent_names)}"
        )

    obs_dims = (
        env.get_observation_dims()
    )

    action_dims = (
        env.get_action_dims()
    )

    action_masks = {
        name: build_action_mask(
            action_dims[name]
        )
        for name in agent_names
    }

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    episode_returns = []
    episode_lengths = []

    # ------------------------------------------------------
    # Episodes
    # ------------------------------------------------------

    for ep in range(num_episodes):

        # Fresh CC4 environment for every episode.
        if ep > 0:
            env = CC4Env(
                red_agent_class=red_agent_class,
            )

        obs_dict, info = env.reset(
            seed=base_seed + ep
        )

        # --------------------------------------------------
        # Communication generated at t-1.
        #
        # None at timestep 0.
        # --------------------------------------------------

        previous_messages = None

        # --------------------------------------------------
        # Episode state
        # --------------------------------------------------

        episode_return = np.zeros(
            NUM_AGENTS,
            dtype=np.float32,
        )

        done = False
        timestep = 0

        # --------------------------------------------------
        # Episode loop
        # --------------------------------------------------

        while (
            not done
            and timestep < EPISODE_LENGTH
        ):

            # ==================================================
            # Pad observations
            # ==================================================

            obs_array = np.zeros(
                (
                    NUM_AGENTS,
                    OBS_DIM,
                ),
                dtype=np.float32,
            )

            for i, name in enumerate(
                agent_names
            ):

                obs_array[i] = (
                    pad_observation(
                        obs_dict[name],
                        obs_dims[name],
                    )
                )

            # ==================================================
            # Centralized critic state
            # ==================================================

            global_obs = (
                obs_array.reshape(-1)
            )

            # ==================================================
            # Actions
            # ==================================================

            actions_dict = {}

            for agent_id, name in enumerate(
                agent_names
            ):

                # --------------------------------------------------
                # Communication available BEFORE this action
                # --------------------------------------------------

                (
                    received_messages,
                    trust_weights,
                ) = get_current_communication_for_agent(
                    ppo=ppo,
                    receiver_id=agent_id,
                    previous_messages=previous_messages,
                )

                # --------------------------------------------------
                # Agent-specific action mask
                # --------------------------------------------------

                mask = action_masks[name]

                # --------------------------------------------------
                # Choose action
                # --------------------------------------------------

                if deterministic:

                    action = (
                        select_action_greedy(
                            ppo=ppo,
                            observation=obs_array[agent_id],
                            action_mask=mask,
                            received_messages=received_messages,
                            trust_weights=trust_weights,
                        )
                    )

                else:

                    action = (
                        select_action_stochastic(
                            ppo=ppo,
                            observation=obs_array[agent_id],
                            action_mask=mask,
                            global_state=global_obs,
                            agent_id=agent_id,
                            received_messages=received_messages,
                            trust_weights=trust_weights,
                        )
                    )

                actions_dict[name] = action

            # ==================================================
            # Generate outgoing communication
            #
            # IMPORTANT:
            #
            # This is generated from the CURRENT observation.
            #
            # It is NOT supplied to the agents' current actions.
            #
            # It becomes available at timestep t+1.
            # ==================================================

            outgoing_vectors = (
                ppo.get_outgoing_messages(
                    obs_array,
                    return_decoded=False,
                )
            )

            # --------------------------------------------------
            # Safety conversion
            # --------------------------------------------------

            if not isinstance(
                outgoing_vectors,
                torch.Tensor,
            ):

                outgoing_vectors = (
                    torch.as_tensor(
                        outgoing_vectors,
                        dtype=torch.float32,
                        device=ppo.device,
                    )
                )

            previous_messages = (
                outgoing_vectors.detach()
            )

            # ==================================================
            # Environment step
            # ==================================================

            (
                next_obs_dict,
                rewards_dict,
                terminated,
                truncated,
                info,
            ) = env.step(
                actions_dict
            )

            # ==================================================
            # Rewards
            # ==================================================

            rewards_arr = np.array(
                [
                    rewards_dict[name]
                    for name in agent_names
                ],
                dtype=np.float32,
            )

            episode_return += (
                rewards_arr
            )

            # ==================================================
            # Done
            # ==================================================

            done = episode_is_done(
                terminated,
                truncated,
            )

            obs_dict = next_obs_dict

            timestep += 1

        # ------------------------------------------------------
        # Episode statistics
        # ------------------------------------------------------

        episode_returns.append(
            float(
                episode_return.sum()
            )
        )

        episode_lengths.append(
            timestep
        )

    return (
        np.asarray(
            episode_returns,
            dtype=np.float32,
        ),
        np.asarray(
            episode_lengths,
            dtype=np.int32,
        ),
    )


# ==========================================================
# Load and evaluate one checkpoint
# ==========================================================

def evaluate_checkpoint(
    checkpoint_path,
    red_agent_names,
    num_episodes,
    deterministic,
):
    """
    Evaluate one checkpoint independently against each Red agent.

    The checkpoint contains:

        model
        actor optimizer
        critic optimizer
        ValueNorm state
        trust state

    MAPPO.load() restores these.

    Trust is then frozen for the entire evaluation.
    """

    print()
    print("=" * 72)
    print(
        f"Checkpoint: "
        f"{os.path.basename(checkpoint_path)}"
    )
    print("=" * 72)

    # ------------------------------------------------------
    # Build environment first.
    #
    # Current MAPPO requires num_targets during construction.
    # ------------------------------------------------------

    reference_env = CC4Env(
        red_agent_class=RED_AGENTS[
            red_agent_names[0]
        ],
    )

    num_targets = (
        reference_env.get_num_targets()
    )

    print(
        f"[MAPPO] Communication target count: "
        f"{num_targets}"
    )

    # ------------------------------------------------------
    # Construct current MAPPO architecture
    # ------------------------------------------------------

    ppo = MAPPO(
        num_targets=num_targets,
    )

    # ------------------------------------------------------
    # Load trained checkpoint
    #
    # This restores:
    #
    #   model
    #   optimizers
    #   ValueNorm
    #   trust state
    # ------------------------------------------------------

    ppo.load(
        checkpoint_path
    )

    # ------------------------------------------------------
    # Frozen evaluation mode
    # ------------------------------------------------------

    ppo.eval()

    results = {}

    # ------------------------------------------------------
    # Evaluate independently against each Red agent
    # ------------------------------------------------------

    for red_name in red_agent_names:

        red_agent_class = (
            RED_AGENTS[red_name]
        )

        print()
        print(
            f"RedAgent: "
            f"{red_agent_class.__name__}"
        )

        returns, lengths = (
            run_episodes(
                ppo=ppo,
                red_agent_class=red_agent_class,
                num_episodes=num_episodes,
                deterministic=deterministic,
            )
        )

        results[red_name] = {
            "mean": float(
                returns.mean()
            ),
            "std": float(
                returns.std()
            ),
            "median": float(
                np.median(returns)
            ),
            "min": float(
                returns.min()
            ),
            "max": float(
                returns.max()
            ),
            "mean_length": float(
                lengths.mean()
            ),
            "n": int(
                num_episodes
            ),
        }

        print(
            f"  mean      = "
            f"{results[red_name]['mean']:9.2f}"
        )

        print(
            f"  std       = "
            f"{results[red_name]['std']:9.2f}"
        )

        print(
            f"  median    = "
            f"{results[red_name]['median']:9.2f}"
        )

        print(
            f"  min       = "
            f"{results[red_name]['min']:9.2f}"
        )

        print(
            f"  max       = "
            f"{results[red_name]['max']:9.2f}"
        )

        print(
            f"  avg length= "
            f"{results[red_name]['mean_length']:9.2f}"
        )

        print(
            f"  episodes  = "
            f"{results[red_name]['n']}"
        )

    # ------------------------------------------------------
    # Combined mean
    # ------------------------------------------------------

    if (
        "random" in results
        and "finite" in results
    ):

        combined_mean = (
            results["random"]["mean"]
            + results["finite"]["mean"]
        ) / 2.0

        print()
        print(
            f"Combined mean = "
            f"{combined_mean:9.2f}"
        )

    return results


# ==========================================================
# Checkpoint discovery
# ==========================================================

def get_checkpoint_list(
    directory,
):
    """
    Return:

        mappo_ep*.pt

    in numeric episode order.
    """

    checkpoints = glob.glob(
        os.path.join(
            directory,
            "mappo_ep*.pt",
        )
    )

    def checkpoint_number(path):

        filename = os.path.basename(
            path
        )

        digits = "".join(
            ch
            for ch in filename
            if ch.isdigit()
        )

        if not digits:
            return -1

        return int(digits)

    return sorted(
        checkpoints,
        key=checkpoint_number,
    )


# ==========================================================
# CLI
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen CC4 MAPPO checkpoint."
        )
    )

    # ------------------------------------------------------
    # Single checkpoint
    # ------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a single .pt checkpoint."
        ),
    )

    # ------------------------------------------------------
    # Checkpoint sweep
    # ------------------------------------------------------

    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help=(
            "Directory containing mappo_ep*.pt checkpoints."
        ),
    )

    # ------------------------------------------------------
    # Episodes
    # ------------------------------------------------------

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help=(
            "Number of evaluation episodes per Red agent."
        ),
    )

    # ------------------------------------------------------
    # Red agent
    # ------------------------------------------------------

    parser.add_argument(
        "--red-agent",
        type=str,
        choices=[
            "random",
            "finite",
            "both",
        ],
        default="both",
        help=(
            "Red opponent to evaluate against."
        ),
    )

    # ------------------------------------------------------
    # Stochastic evaluation
    # ------------------------------------------------------

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Sample actions instead of using greedy argmax."
        ),
    )

    args = parser.parse_args()

    # ======================================================
    # Validate checkpoint arguments
    # ======================================================

    if (
        args.checkpoint is None
        and args.sweep is None
    ):

        parser.error(
            "Provide either --checkpoint or --sweep."
        )

    if (
        args.checkpoint is not None
        and args.sweep is not None
    ):

        parser.error(
            "Use either --checkpoint or --sweep, "
            "not both."
        )

    # ======================================================
    # Validate episode count
    # ======================================================

    if args.episodes <= 0:

        parser.error(
            "--episodes must be greater than zero."
        )

    # ======================================================
    # Red agent selection
    # ======================================================

    if args.red_agent == "both":

        red_agent_names = [
            "random",
            "finite",
        ]

    else:

        red_agent_names = [
            args.red_agent
        ]

    # ======================================================
    # Evaluation mode
    # ======================================================

    deterministic = (
        not args.stochastic
    )

    # ======================================================
    # Build checkpoint list
    # ======================================================

    if args.checkpoint is not None:

        if not os.path.isfile(
            args.checkpoint
        ):

            parser.error(
                "Checkpoint does not exist: "
                f"{args.checkpoint}"
            )

        checkpoints = [
            args.checkpoint
        ]

    else:

        checkpoints = (
            get_checkpoint_list(
                args.sweep
            )
        )

        if not checkpoints:

            parser.error(
                "No mappo_ep*.pt checkpoints found in: "
                f"{args.sweep}"
            )

    # ======================================================
    # Header
    # ======================================================

    print()
    print("=" * 72)
    print("CC4 MAPPO EVALUATION")
    print("=" * 72)

    print(
        "Architecture : GNN + Attention"
    )

    print(
        "Communication: V2 structured"
    )

    print(
        "Trust        : checkpoint state, frozen"
    )

    print(
        "PPO updates  : disabled"
    )

    print(
        "Ground truth : disabled"
    )

    print(
        "Mode         : "
        + (
            "greedy"
            if deterministic
            else "stochastic"
        )
    )

    print(
        "Episodes     : "
        f"{args.episodes} per Red agent"
    )

    print("=" * 72)

    # ======================================================
    # Evaluate
    # ======================================================

    for checkpoint in checkpoints:

        evaluate_checkpoint(
            checkpoint_path=checkpoint,
            red_agent_names=red_agent_names,
            num_episodes=args.episodes,
            deterministic=deterministic,
        )


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()
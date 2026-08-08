"""
train.py

Training script for MAPPO on CAGE Challenge 4 (CC4).

Communication
-------------

At timestep t:

    previous messages
            +
       trust weights
            |
            v
          MAPPO
            |
       +----+----+
       |         |
     action    message
       |         |
       v         v
     CybORG   communication
       |
       v
    rewards

The message generated at timestep t becomes available to the
other agents at timestep t+1. This one-step delay prevents an agent
from receiving information generated from the same observation it is
currently using to select its action.

Differentiable communication (new)
-----------------------------------

Alongside previous_messages (used for ACTING during rollout, under
no_grad -- unchanged), we now also track previous_obs_array: the raw
multi-agent observation array that PRODUCED those messages. This gets
stored per-row as communication_source_obs, so mappo.py.update() can
re-run it through the CURRENT sender network with gradients enabled and
keep the decoder/encoder in the training graph. communication_valid is
False for the first row of each episode, where no previous observation
exists yet.

Trust is updated only when a valid training-side ground-truth
representation is available (see env.py.get_ground_truth). Ground
truth is NEVER passed to the Blue agents.
"""

import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .env import CC4Env
from .buffer import MAPPOBuffer
from .mappo import MAPPO
from .communication.evaluator import MessageEvaluator

from CybORG.Agents import (
    SleepAgent,  # not using bhai
    RandomSelectRedAgent,
    FiniteStateRedAgent,
)

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    ACTION_DIM,
    EPISODE_LENGTH,
    TOTAL_EPISODES,
    ROLLOUT_STEPS,
    SEED,
    PRINT_EVERY,
    CURRICULUM_SCHEDULE,
    SAVE_EVERY,
    CHECKPOINT_DIR,
    LOG_DIR,
    CURRICULUM_SWITCH_EPISODE,
)


RED_AGENT_MAP = {
    "RandomSelectRedAgent": RandomSelectRedAgent,
    "FiniteStateRedAgent": FiniteStateRedAgent,
}


############################################################
# Seeding
############################################################

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


############################################################
# Padding helpers
############################################################

def pad_observation(obs, real_dim):

    padded = np.zeros(OBS_DIM, dtype=np.float32)
    padded[:real_dim] = obs
    return padded


def build_action_mask(real_action_dim):

    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[:real_action_dim] = True
    return mask


############################################################
# Episode boundary helper
############################################################

def episode_is_done(terminated, truncated):

    if "__all__" in terminated or "__all__" in truncated:

        return (
            terminated.get("__all__", False)
            or truncated.get("__all__", False)
        )

    return (
        all(terminated.values())
        or all(truncated.values())
    )


############################################################
# Curriculum
############################################################

def get_curriculum_stage(episode):
    """
    Progressive probabilistic curriculum.
    """

    probability_finite = 1.0

    for max_episode, p in CURRICULUM_SCHEDULE:

        if episode < max_episode:
            probability_finite = p
            break

    if random.random() < probability_finite:
        return FiniteStateRedAgent

    return RandomSelectRedAgent


############################################################
# Communication helpers
############################################################

def get_current_communication_for_agent(
    ppo,
    receiver_id,
    previous_messages,
):
    """
    Return the messages and trust values available to one agent
    before it selects its action. previous_messages are from the
    previous timestep, so there is no information leakage.
    """

    if previous_messages is None:
        return None, None

    received_messages = ppo.get_messages_for_agent(
        receiver_id=receiver_id,
        messages=previous_messages,
    )

    trust_weights = ppo.get_trust_for_agent(
        receiver_id=receiver_id,
    )

    return received_messages, trust_weights


############################################################
# Ground-truth hook
############################################################

def get_ground_truth_for_message(
    env,
    sender_id,
    receiver_id,
    message,
    previous_info,
    current_info,
):
    """
    Obtain training-side ground truth for evaluating a message.

    Supported environment interfaces (checked in order):

        env.get_message_ground_truth(...)
        env.get_ground_truth(...)

    Returns None if neither exists -- this deliberately prevents
    training trust using fabricated labels.
    """

    if hasattr(env, "get_message_ground_truth"):

        return env.get_message_ground_truth(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message=message,
            previous_info=previous_info,
            current_info=current_info,
        )

    if hasattr(env, "get_ground_truth"):

        return env.get_ground_truth(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message=message,
            previous_info=previous_info,
            current_info=current_info,
        )

    return None


############################################################
# Evaluate outgoing messages and update trust
############################################################

def evaluate_and_update_trust(
    ppo,
    evaluator,
    env,
    outgoing_structured_messages,
    previous_info,
    current_info,
):
    """
    Evaluate messages after the environment step and update trust.

    outgoing_structured_messages: list of StructuredMessage, index = sender.

    Trust convention: trust(sender, receiver) means how much receiver
    trusts sender. Each sender's message is evaluated separately for
    every receiver. If ground truth is unavailable, no trust update
    is performed.
    """

    if (
        outgoing_structured_messages is None
        or len(outgoing_structured_messages) != NUM_AGENTS
    ):
        return

    for sender_id in range(NUM_AGENTS):

        message = outgoing_structured_messages[sender_id]

        if message is None:
            continue

        if hasattr(message, "is_empty") and message.is_empty():
            continue

        for receiver_id in range(NUM_AGENTS):

            if sender_id == receiver_id:
                continue

            ground_truth = get_ground_truth_for_message(
                env=env,
                sender_id=sender_id,
                receiver_id=receiver_id,
                message=message,
                previous_info=previous_info,
                current_info=current_info,
            )

            if ground_truth is None:
                continue

            evaluation = evaluator.evaluate(
                message=message,
                ground_truth=ground_truth,
                previous_state=previous_info,
                current_state=current_info,
            )

            quality = evaluator.confidence_adjusted_score(evaluation)

            ppo.update_trust(
                sender=sender_id,
                receiver=receiver_id,
                message_quality=quality,
            )
            # print(
            #     f"[TRUST] "
            #     f"{sender_id}->{receiver_id} "
            #     f"quality={quality:.3f} "
            #     f"trust={ppo.get_trust(sender_id, receiver_id):.3f}"
            # )


############################################################
# Main training loop
############################################################

def train():

    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ########################################################
    # Environment
    ########################################################

    current_red_agent = get_curriculum_stage(0)

    env = CC4Env(red_agent_class=current_red_agent)

    agent_names = sorted(env.possible_agents)

    assert len(agent_names) == NUM_AGENTS, (
        f"Expected {NUM_AGENTS} blue agents, "
        f"found {len(agent_names)}: {agent_names}"
    )

    obs_dims = env.get_observation_dims()
    action_dims = env.get_action_dims()

    action_masks = {
        name: build_action_mask(action_dims[name])
        for name in agent_names
    }

    ########################################################
    # Initial reset
    ########################################################

    

    obs_dict, info = env.reset(seed=SEED)

########################################################
# Communication target vocabulary
########################################################

    num_targets = env.get_num_targets()

    print(
        f"[MAPPO] Communication target count: {num_targets}"
    )

    ########################################################
    # Agent / Buffer
    ########################################################

    ppo = MAPPO(
        num_targets=num_targets
    )

    buffer = MAPPOBuffer()
    evaluator = MessageEvaluator()

    ########################################################
    # Runtime communication state
    ########################################################

    previous_messages = None
    previous_obs_array = None



    episode_return = np.zeros(NUM_AGENTS, dtype=np.float32)
    episode_returns_log = []
    episode_count = 0
    update_count = 0

    ########################################################
    # Training History
    ########################################################

    episode_return_history = []
    actor_loss_history = []
    critic_loss_history = []
    entropy_history = []

    previous_info = info

    total_timesteps = TOTAL_EPISODES * EPISODE_LENGTH

    start_time = time.time()

    ########################################################
    # Rollout / Update Loop
    ########################################################

    for t in range(1, total_timesteps + 1):

        ####################################################
        # Build padded local + global observations
        ####################################################

        obs_array = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)

        for i, name in enumerate(agent_names):
            obs_array[i] = pad_observation(obs_dict[name], obs_dims[name])

        global_obs = obs_array.reshape(-1)

        ####################################################
        # Messages available BEFORE acting
        ####################################################

        current_received_messages = (
            np.zeros(
                (NUM_AGENTS, NUM_AGENTS, ppo.communication.message_dim),
                dtype=np.float32,
            )
            if previous_messages is not None
            else None
        )

        current_trust_weights = (
            np.zeros((NUM_AGENTS, NUM_AGENTS), dtype=np.float32)
            if previous_messages is not None
            else None
        )

        ####################################################
        # Act
        ####################################################

        actions_dict = {}
        actions_arr = np.zeros(NUM_AGENTS, dtype=np.int64)
        log_probs_arr = np.zeros(NUM_AGENTS, dtype=np.float32)
        values_arr = np.zeros(NUM_AGENTS, dtype=np.float32)
        masks_arr = np.zeros((NUM_AGENTS, ACTION_DIM), dtype=bool)

        for i, name in enumerate(agent_names):

            mask = action_masks[name]
            masks_arr[i] = mask

            ################################################
            # Communication available to this receiver
            ################################################

            received_messages, trust_weights = get_current_communication_for_agent(
                ppo=ppo,
                receiver_id=i,
                previous_messages=previous_messages,
            )

            ################################################
            # Save communication state for the buffer
            ################################################

            if received_messages is not None:

                current_received_messages[i] = (
                    received_messages.detach().cpu().numpy()
                )
                current_trust_weights[i] = (
                    trust_weights.detach().cpu().numpy()
                )

            ################################################
            # Select action
            ################################################

            action, log_prob, value, entropy = ppo.select_action(
                observation=obs_array[i],
                action_mask=mask,
                global_state=global_obs,
                agent_id=i,
                received_messages=received_messages,
                trust_weights=trust_weights,
            )

            actions_dict[name] = action
            actions_arr[i] = action
            log_probs_arr[i] = log_prob.item()
            values_arr[i] = 0.0 if value is None else value.item()

        ####################################################
        # Generate outgoing structured messages
        #
        # Generated from the CURRENT observation, but not delivered
        # until t+1. Still no_grad -- this is behavior, not training.
        ####################################################

        (
            outgoing_vectors,
            outgoing_structured_messages,
            communication_field_ids,
            communication_log_probs,
            communication_entropies,
        ) = ppo.get_outgoing_messages(
            obs_array,
            return_decoded=True,
        )

        ####################################################
        # Step environment
        ####################################################

        (
            next_obs_dict,
            rewards_dict,
            terminated,
            truncated,
            info,
        ) = env.step(actions_dict)

        rewards_arr = np.array(
            [rewards_dict[name] for name in agent_names],
            dtype=np.float32,
        )

        done_flag = episode_is_done(terminated, truncated)

        dones_arr = np.full(NUM_AGENTS, float(done_flag), dtype=np.float32)

        ####################################################
        # Evaluate message quality / update trust
        ####################################################

        evaluate_and_update_trust(
            ppo=ppo,
            evaluator=evaluator,
            env=env,
            outgoing_structured_messages=outgoing_structured_messages,
            previous_info=previous_info,
            current_info=info,
        )

        ####################################################
        # Store transition
        #
        # communication_source_obs/communication_valid are built from
        # previous_obs_array -- the obs that produced the messages
        # actually received BEFORE this step's action, mirroring
        # current_received_messages exactly.
        ####################################################

        communication_source = (
            previous_obs_array
            if previous_obs_array is not None
            else np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)
        )

        buffer.store(
            obs=obs_array,
            global_obs=global_obs,
            actions=actions_arr,
            log_probs=log_probs_arr,
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            action_masks=masks_arr,
            received_messages=current_received_messages,
            trust_weights=current_trust_weights,
            communication_source_obs=communication_source,
            communication_valid=(previous_obs_array is not None),

            communication_field_ids=communication_field_ids,
            communication_log_probs=communication_log_probs,
            communication_entropies=communication_entropies,
        )

        ####################################################
        # Newly generated messages/obs become available at t+1
        ####################################################

        previous_messages = outgoing_vectors.detach()
        previous_obs_array = obs_array

        ####################################################
        # Update evaluator state
        ####################################################

        previous_info = info
        episode_return += rewards_arr
        obs_dict = next_obs_dict

        ####################################################
        # Episode boundary
        ####################################################

        if done_flag:

            episode_count += 1

            current_red_agent = get_curriculum_stage(episode_count)
            env = CC4Env(red_agent_class=current_red_agent)

            team_return = episode_return.sum()
            episode_returns_log.append(team_return)
            episode_return_history.append(team_return)

            if episode_count % PRINT_EVERY == 0:

                recent = episode_returns_log[-PRINT_EVERY:]
                mean_return = float(np.mean(recent))
                elapsed = time.time() - start_time

                print(
                    f"[episode {episode_count:6d}] "
                    f"team_return={mean_return:8.2f}  "
                    f"updates={update_count:5d}  "
                    f"elapsed={elapsed:7.1f}s  "
                    f"RedAgent={current_red_agent.__name__}"
                )

            if episode_count % SAVE_EVERY == 0:

                ckpt_path = os.path.join(
                    CHECKPOINT_DIR, f"mappo_ep{episode_count}.pt"
                )
                ppo.save(ckpt_path)

            episode_return[:] = 0.0

            # ------------------------------------------------
            # Reset communication memory -- no message and no
            # source observation carries over from the previous
            # episode.
            # ------------------------------------------------

            previous_messages = None
            previous_obs_array = None

            obs_dict, info = env.reset(seed=SEED + episode_count)
            previous_info = info

        ####################################################
        # PPO Update
        ####################################################

        if buffer.is_full():

            last_obs_array = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)

            for i, name in enumerate(agent_names):
                last_obs_array[i] = pad_observation(obs_dict[name], obs_dims[name])

            last_global_obs = last_obs_array.reshape(-1)

            last_values = ppo.get_value(last_global_obs).cpu().numpy()

            buffer.compute_advantages(last_values)

            ppo.train()
            stats = ppo.update(buffer)

            actor_loss_history.append(stats["actor_loss"])
            critic_loss_history.append(stats["critic_loss"])
            entropy_history.append(stats["entropy"])

            ppo.eval()

            update_count += 1

            print(
                f"  -> update {update_count:5d}  "
                f"actor_loss={stats['actor_loss']:.4f}  "
                f"critic_loss={stats['critic_loss']:.4f}  "
                f"entropy={stats['entropy']:.4f}"
            )

            buffer.clear()

    ########################################################
    # Final checkpoint
    ########################################################

    final_path = os.path.join(CHECKPOINT_DIR, "mappo_final.pt")
    ppo.save(final_path)

    print(f"Training complete. Final checkpoint: {final_path}")

    ########################################################
    # Plots
    ########################################################

    plt.figure(figsize=(10, 5))
    plt.plot(episode_return_history)
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    plt.savefig("evaluation/episode_return.png")

    plt.figure(figsize=(10, 5))
    plt.plot(actor_loss_history)
    plt.title("Actor Loss")
    plt.xlabel("PPO Update")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("evaluation/actor_loss.png")

    plt.figure(figsize=(10, 5))
    plt.plot(critic_loss_history)
    plt.title("Critic Loss")
    plt.xlabel("PPO Update")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("evaluation/critic_loss.png")

    plt.figure(figsize=(10, 5))
    plt.plot(entropy_history)
    plt.title("Entropy")
    plt.xlabel("PPO Update")
    plt.ylabel("Entropy")
    plt.grid()
    plt.savefig("evaluation/entropy.png")


if __name__ == "__main__":
    train()
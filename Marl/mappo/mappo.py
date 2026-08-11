"""
mappo.py

Multi-Agent PPO (MAPPO) implementation for CC4.

Contains
--------
- Shared Actor
- Centralized Critic
- Action Selection
- Value Prediction
- Structured Communication
- Dynamic Trust
- Save / Load
- PPO Update

Structured communication training
----------------------------------

Rollout communication is generated under no_grad(), which is correct.

The rollout buffer stores:

    communication_source_obs

During PPO optimization, messages are reconstructed from those
observations through the CURRENT sender network:

    source observation
            |
            v
      sender actor
            |
            v
       decoder
            |
            v
       encoder
            |
            v
 communication vector
            |
            v
    receiver attention
            |
            v
        PPO loss

Therefore gradients can flow through the complete communication
pipeline.

The detached ``received_messages`` stored in the buffer are retained
for compatibility/debugging, but are NOT used as the differentiable
communication source during PPO optimization.
"""

import numpy as np
import torch
import torch.nn.functional as F

from torch.distributions import Categorical

from .gnn_attention import MAPPOModel, COMMUNICATION_DIM

from .value_norm import ValueNorm

from .config import (
    DEVICE,
    LEARNING_RATE,
    ACTOR_LEARNING_RATE,
    USE_VALUE_NORM,
    CRITIC_LEARNING_RATE,
    MAX_GRAD_NORM,
    UPDATE_EPOCHS,
    MINIBATCH_SIZE,
    PPO_CLIP,
    VALUE_LOSS_COEF,
    ENTROPY_COEF,
    NORMALIZE_ADVANTAGES,
    NUM_AGENTS,
    OBS_DIM,
)


class MAPPO:

    # ==========================================================
    # Initialization
    # ==========================================================

    def __init__(self,num_targets):

        self.device = DEVICE

        # ------------------------------------------------------
        # Networks
        # ------------------------------------------------------

        # self.model = MAPPOModel().to(
        #     self.device,
        #     num_targets=YOUR_TARGET_COUNT
        # )


        self.model = MAPPOModel(
            num_targets=num_targets
        ).to(self.device)
        self.actor = self.model.actor

        self.critic = self.model.critic

        # ------------------------------------------------------
        # Structured communication
        # ------------------------------------------------------

        self.communication = (
            self.actor.communication
        )

        # ------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LEARNING_RATE,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LEARNING_RATE,
        )

        # ------------------------------------------------------
        # Gradient clipping
        # ------------------------------------------------------

        self.max_grad_norm = MAX_GRAD_NORM

        # ------------------------------------------------------
        # Value normalization
        # ------------------------------------------------------

        self.value_norm = None

        if USE_VALUE_NORM:

            self.value_norm = ValueNorm(
                device=self.device,
            )

        # ------------------------------------------------------
        # Runtime communication state
        # ------------------------------------------------------

        self.previous_messages = None

        self.current_messages = None

        self.current_decoded_messages = None

        self.trust_weights = None

    # ==========================================================
    # Tensor Utilities
    # ==========================================================

    def _to_tensor(
        self,
        value,
        dtype=torch.float32,
    ):
        """
        Convert numpy/list/tensor to a device tensor.
        """

        if isinstance(value, torch.Tensor):

            return value.to(
                device=self.device,
                dtype=dtype,
            )

        return torch.tensor(
            value,
            dtype=dtype,
            device=self.device,
        )

    # ==========================================================
    # Action Mask
    # ==========================================================

    def _apply_action_mask(
        self,
        logits,
        action_masks=None,
    ):
        """
        Apply action masks.

        True  = valid action
        False = invalid action
        """

        if action_masks is None:

            return logits

        return logits.masked_fill(
            ~action_masks.bool(),
            -1e10,
        )

    # ==========================================================
    # Action Selection
    # ==========================================================

    @torch.no_grad()
    def select_action(
        self,
        observation,
        action_mask=None,
        global_state=None,
        agent_id=None,
        received_messages=None,
        trust_weights=None,
    ):
        """
        Select an action using the shared actor.

        Existing return interface is preserved:

            action
            log_prob
            value
            entropy

        Communication can be supplied through:

            received_messages
            trust_weights
        """

        observation = self._to_tensor(
            observation,
            dtype=torch.float32,
        )

        # ------------------------------------------------------
        # Actor
        # ------------------------------------------------------

        logits = self.actor(
            observation,
            received_messages=received_messages,
            trust_weights=trust_weights,
        )

        # ------------------------------------------------------
        # Action mask
        # ------------------------------------------------------

        if action_mask is not None:

            action_mask = self._to_tensor(
                action_mask,
                dtype=torch.bool,
            )

            logits = self._apply_action_mask(
                logits,
                action_mask,
            )

        # ------------------------------------------------------
        # Distribution
        # ------------------------------------------------------

        distribution = Categorical(
            logits=logits
        )

        action = distribution.sample()

        log_prob = distribution.log_prob(
            action
        )

        entropy = distribution.entropy()

        # ------------------------------------------------------
        # Centralized critic
        # ------------------------------------------------------

        value = None

        if global_state is not None:

            global_state = self._to_tensor(
                global_state,
                dtype=torch.float32,
            )

            values = self.critic(
                global_state
            )

            if agent_id is not None:

                value = values[agent_id]

                if self.value_norm is not None:

                    value = (
                        self.value_norm.denormalize(
                            value
                        )
                    )

            else:

                value = values

                if self.value_norm is not None:

                    value = (
                        self.value_norm.denormalize(
                            value
                        )
                    )

        return (
            action.item(),
            log_prob.detach(),
            None
            if value is None
            else value.detach(),
            entropy.detach(),
        )

    # ==========================================================
    # Generate Outgoing Structured Message
    # ==========================================================

    @torch.no_grad()
    def get_outgoing_message(
        self,
        observation,
        return_decoded=False,
    ):
        """
        Generate one agent's outgoing communication vector.
        """

        observation = self._to_tensor(
            observation,
            dtype=torch.float32,
        )

        local_hidden = (
            self.actor.get_local_hidden(
                observation
            )
        )

        (
            field_ids,
            message_log_probs,
            message_entropies,
            communication_vector,
        ) = self.communication.generate_message(
            local_hidden
        )

        communication_vector = (
            communication_vector.detach()
        )

        self.current_messages = (
            communication_vector
        )

        if return_decoded:

            structured_message = (
                self.actor.communication.decoder.decode(
                    local_hidden
                )
            )

            self.current_decoded_messages = (
                structured_message
            )

            return (
                communication_vector,
                field_ids,
                message_log_probs,
                message_entropies,
            )

        return communication_vector

    # ==========================================================
    # Generate Messages For All Agents
    # ==========================================================

    @torch.no_grad()
    def get_outgoing_messages(
        self,
        observations,
        return_decoded=False,
    ):
        """
        Generate communication for every Blue agent.

        observations:

            [NUM_AGENTS, OBS_DIM]

        messages:

            [NUM_AGENTS, COMMUNICATION_DIM]
        """

        observations = self._to_tensor(
            observations,
            dtype=torch.float32,
        )

        if observations.ndim != 2:

            raise ValueError(
                "observations must have shape "
                "[NUM_AGENTS, OBS_DIM]"
            )

        if observations.shape[0] != NUM_AGENTS:

            raise ValueError(
                f"Expected {NUM_AGENTS} observations, "
                f"got {observations.shape[0]}"
            )

        local_hidden = (
            self.actor.get_local_hidden(
                observations
            )
        )

        (
            field_ids,
            message_log_probs,
            message_entropies,
            messages,
        ) = self.communication.generate_message(
            local_hidden
        )

        # ------------------------------------------------------
        # Convert v2 communication dictionaries to [N, 7]
        # arrays for the rollout buffer.
        # ------------------------------------------------------

        field_order = (
            "event_type",
            "target_type",
            "threat_level",
            "status",
            "priority",
            "confidence",
            "target_id",
        )

        communication_field_ids = torch.stack(
            [
                field_ids[name]
                for name in field_order
            ],
            dim=-1,
        )

        communication_log_probs = torch.stack(
            [
                message_log_probs[name]
                for name in field_order
            ],
            dim=-1,
        )

        communication_entropies = torch.stack(
            [
                message_entropies[name]
                for name in field_order
            ],
            dim=-1,
        )

        messages = messages.detach()

        self.current_messages = messages

        if return_decoded:

            decoded_messages = []

            for i in range(NUM_AGENTS):

                decoded_messages.append(
                    self.actor.communication.decoder.decode(
                        local_hidden[i]
                    )
                )

            self.current_decoded_messages = (
                decoded_messages
            )

            # return (
            #     messages,
            #     decoded_messages,
            # )
            return (
                messages,
                decoded_messages,
                communication_field_ids.detach().cpu().numpy(),
                communication_log_probs.detach().cpu().numpy(),
                communication_entropies.detach().cpu().numpy(),
            )


        # return messages
        return (
            messages,
            communication_field_ids.detach().cpu().numpy(),
            communication_log_probs.detach().cpu().numpy(),
            communication_entropies.detach().cpu().numpy(),
        )

    # ==========================================================
    # Build Communication Matrix
    # ==========================================================

    @torch.no_grad()
    def build_received_messages(
        self,
        communication_vectors,
    ):
        """
        Convert:

            [sender, D]

        into:

            [receiver, sender, D]
        """

        communication_vectors = (
            self._to_tensor(
                communication_vectors,
                dtype=torch.float32,
            )
        )

        return (
            self.communication.create_message_matrix(
                communication_vectors
            )
        )

    # ==========================================================
    # Apply Trust To Communication
    # ==========================================================

    @torch.no_grad()
    def get_trusted_messages(
        self,
        communication_vectors,
    ):
        """
        Apply current trust to communication vectors.
        """

        communication_vectors = (
            self._to_tensor(
                communication_vectors,
                dtype=torch.float32,
            )
        )

        return (
            self.communication.apply_trust(
                communication_vectors
            )
        )

    # ==========================================================
    # Trust Matrix
    # ==========================================================

    def get_trust_matrix(self):

        return (
            self.communication.get_trust_matrix()
        )

    # ==========================================================
    # Individual Trust
    # ==========================================================

    def get_trust(
        self,
        sender,
        receiver,
    ):

        return self.communication.get_trust(
            sender,
            receiver,
        )

    # ==========================================================
    # Update Trust
    # ==========================================================

    def update_trust(
        self,
        sender,
        receiver,
        message_quality,
        weight=1.0,
    ):

        return (
            self.communication.update_trust(
                sender=sender,
                receiver=receiver,
                message_quality=message_quality,
                weight=weight,
            )
        )

    # ==========================================================
    # Update Trust Matrix
    # ==========================================================

    def update_trust_matrix(
        self,
        evaluations,
    ):

        return (
            self.communication.update_trust_matrix(
                evaluations
            )
        )

    # ==========================================================
    # Reset Communication
    # ==========================================================

    def reset_communication(self):

        self.previous_messages = None

        self.current_messages = None

        self.current_decoded_messages = None

        self.trust_weights = None

        self.communication.reset_trust()

    # ==========================================================
    # Previous Messages
    # ==========================================================

    def set_previous_messages(
        self,
        messages,
    ):

        if messages is None:

            self.previous_messages = None

            return

        messages = self._to_tensor(
            messages,
            dtype=torch.float32,
        )

        self.previous_messages = (
            messages.detach()
        )

    # ==========================================================
    # Messages For One Receiver
    # ==========================================================

    def get_messages_for_agent(
        self,
        receiver_id,
        messages=None,
    ):
        """
        Return:

            [NUM_AGENTS, COMMUNICATION_DIM]

        for one receiver.
        """

        if messages is None:

            messages = self.previous_messages

        if messages is None:

            return None

        messages = self._to_tensor(
            messages,
            dtype=torch.float32,
        )

        if messages.ndim != 2:

            raise ValueError(
                "messages must have shape "
                "[NUM_AGENTS, COMMUNICATION_DIM]"
            )

        if not (
            0 <= receiver_id < NUM_AGENTS
        ):

            raise ValueError(
                f"receiver_id must be in "
                f"[0, {NUM_AGENTS - 1}]"
            )

        receiver_messages = (
            messages.clone()
        )

        receiver_messages[
            receiver_id
        ] = 0.0

        return receiver_messages

    # ==========================================================
    # Trust For Receiver
    # ==========================================================

    def get_trust_for_agent(
        self,
        receiver_id,
    ):
        """
        Return:

            trust[sender]

        for a fixed receiver.
        """

        trust_matrix = (
            self.get_trust_matrix()
        )

        trust_for_receiver = (
            trust_matrix[:, receiver_id]
        )

        return trust_for_receiver.to(
            device=self.device,
            dtype=torch.float32,
        )

    # ==========================================================
    # Critic
    # ==========================================================

    @torch.no_grad()
    def get_value(
        self,
        global_state,
    ):

        global_state = self._to_tensor(
            global_state,
            dtype=torch.float32,
        )

        values = self.critic(
            global_state
        )

        if self.value_norm is not None:

            values = (
                self.value_norm.denormalize(
                    values
                )
            )

        return values

    # ==========================================================
    # Train / Eval
    # ==========================================================

    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()

    # ==========================================================
    # Actor Forward
    # ==========================================================

    def actor_forward(
        self,
        observations,
        action_masks=None,
        received_messages=None,
        trust_weights=None,
    ):
        """
        Standard actor forward.

        This remains available for compatibility.
        """

        logits = self.actor(
            observations,
            received_messages=received_messages,
            trust_weights=trust_weights,
        )

        logits = self._apply_action_mask(
            logits,
            action_masks,
        )

        return logits

    # ==========================================================
    # Critic Forward
    # ==========================================================

    def critic_forward(
        self,
        global_states,
    ):

        return self.critic(
            global_states
        )

    # ==========================================================
    # DIFFERENTIABLE COMMUNICATION RECONSTRUCTION
    # ==========================================================

    def _reconstruct_received_messages(
        self,
        communication_source_obs,
        communication_field_ids,
        agent_ids,
        communication_valid=None,
        trust_weights=None,
    ):

        if communication_source_obs is None:

            return None, trust_weights

        source_obs = communication_source_obs

        # ----------------------------------------------------------
        # Shape checks
        # ----------------------------------------------------------

        if source_obs.ndim != 3:

            raise ValueError(
                "communication_source_obs must have shape "
                "[B, NUM_AGENTS, OBS_DIM]. "
                f"Got {tuple(source_obs.shape)}"
            )

        batch_size = source_obs.shape[0]

        if source_obs.shape[1] != NUM_AGENTS:

            raise ValueError(
                f"Expected {NUM_AGENTS} sender observations, "
                f"got {source_obs.shape[1]}"
            )

        if source_obs.shape[2] != OBS_DIM:

            raise ValueError(
                f"Expected OBS_DIM={OBS_DIM}, "
                f"got {source_obs.shape[2]}"
            )

        # ----------------------------------------------------------
        # Agent IDs
        # ----------------------------------------------------------

        agent_ids = agent_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        if agent_ids.ndim != 1:

            agent_ids = agent_ids.reshape(
                -1
            )

        if agent_ids.shape[0] != batch_size:

            raise ValueError(
                "agent_ids batch dimension does not match "
                "communication_source_obs."
            )

        if torch.any(
            agent_ids < 0
        ) or torch.any(
            agent_ids >= NUM_AGENTS
        ):

            raise ValueError(
                "agent_ids contains an invalid agent index."
            )

        # ----------------------------------------------------------
        # Generate sender hidden representations
        #
        # [B, N, OBS]
        #       ↓
        # [B*N, OBS]
        #
        # This is the CURRENT sender network.
        #
        # No detach()
        # No no_grad()
        # ----------------------------------------------------------

        sender_obs = source_obs.reshape(
            batch_size * NUM_AGENTS,
            OBS_DIM,
        )

        # sender_hidden = (
        #     self.actor.get_local_hidden(
        #         sender_obs
        #     )
        # )

        # ----------------------------------------------------------
        # Structured decoder + encoder
        #
        # [B*N, hidden]
        #       ↓
        # [B*N, COMMUNICATION_DIM]
        # ----------------------------------------------------------

        # (
        #     sender_field_ids,
        #     sender_log_probs,
        #     sender_entropies,
        #     sender_messages,
        # ) = self.communication.generate_message(
        #     sender_hidden
        # )

        # sender_messages = self.communication.encoder.encode_from_ids(communication_field_ids)
        sender_field_ids = {
            "event_type": communication_field_ids[:, :, 0],
            "target_type": communication_field_ids[:, :, 1],
            "threat_level": communication_field_ids[:, :, 2],
            "status": communication_field_ids[:, :, 3],
            "priority": communication_field_ids[:, :, 4],
            "confidence": communication_field_ids[:, :, 5],
            "target_id": communication_field_ids[:, :, 6],
        }

        sender_messages = self.communication.encoder.encode_from_ids(
            sender_field_ids
        )

        # ----------------------------------------------------------
        # Restore sender dimension
        #
        # [B*N, D]
        #       ↓
        # [B, N, D]
        # ----------------------------------------------------------

        sender_messages = (
            sender_messages.reshape(
                batch_size,
                NUM_AGENTS,
                COMMUNICATION_DIM,
            )
        )

        # ----------------------------------------------------------
        # Every receiver initially sees every sender.
        #
        # [B, N_sender, D]
        #
        # No receiver expansion is required here because each PPO
        # sample already represents ONE receiver.
        # ----------------------------------------------------------

        received_messages = sender_messages

        # ----------------------------------------------------------
        # Remove self-communication.
        #
        # For sample b:
        #
        #     receiver = agent_ids[b]
        #
        # Therefore:
        #
        #     received_messages[b, receiver] = 0
        # ----------------------------------------------------------

        batch_indices = torch.arange(
            batch_size,
            device=self.device,
        )

        received_messages = (
            received_messages.clone()
        )

        received_messages[
            batch_indices,
            agent_ids,
            :
        ] = 0.0

        # ----------------------------------------------------------
        # Trust
        #
        # trust_weights is ALREADY:
        #
        #     [B, sender]
        #
        # because update() flattened:
        #
        #     [T, receiver, sender]
        #
        # into:
        #
        #     [T*N, sender]
        #
        # Therefore DO NOT expand it again.
        # ----------------------------------------------------------

        receiver_trust = None

        if trust_weights is not None:

            if trust_weights.ndim != 2:

                raise ValueError(
                    "trust_weights must have shape "
                    "[B, NUM_AGENTS]. "
                    f"Got {tuple(trust_weights.shape)}"
                )

            if trust_weights.shape[0] != batch_size:

                raise ValueError(
                    "trust_weights batch dimension does not "
                    "match communication_source_obs."
                )

            if trust_weights.shape[1] != NUM_AGENTS:

                raise ValueError(
                    f"Expected trust for {NUM_AGENTS} senders, "
                    f"got {trust_weights.shape[1]}"
                )

            receiver_trust = torch.clamp(
                trust_weights,
                0.0,
                1.0,
            )

        # ----------------------------------------------------------
        # Episode boundary handling
        # ----------------------------------------------------------

        if communication_valid is not None:

            valid = communication_valid

            if valid.ndim > 1:

                valid = valid.reshape(
                    batch_size
                )

            valid = valid.bool()

            if valid.shape[0] != batch_size:

                raise ValueError(
                    "communication_valid must have shape [B]."
                )

            # Invalid communication samples receive zero messages.
            received_messages = (
                received_messages
                * valid.to(
                    dtype=received_messages.dtype
                ).view(
                    batch_size,
                    1,
                    1,
                )
            )

            if receiver_trust is not None:

                receiver_trust = (
                    receiver_trust
                    * valid.to(
                        dtype=receiver_trust.dtype
                    ).view(
                        batch_size,
                        1,
                    )
                )

        return (
            received_messages,
            receiver_trust,
        )

    # ==========================================================
    # Evaluate Actions
    # ==========================================================

    def evaluate_actions(
        self,
        observations,
        global_states,
        actions,
        agent_ids,
        action_masks=None,
        received_messages=None,
        trust_weights=None,
        communication_source_obs=None,
        communication_field_ids=None,
        communication_valid=None,
    ):
        """
        Evaluate actions during PPO optimization.

        Communication modes
        -------------------

        1. Differentiable mode:

            communication_source_obs is supplied.

            Messages are regenerated through:

                sender actor
                    ->
                decoder
                    ->
                encoder
                    ->
                receiver attention

            This is the mode used by the new PPO update.

        2. Legacy mode:

            received_messages is supplied.

            These are treated as already-generated tensors.

        3. No communication:

            both are None.

        Returns
        -------

        log_probs
        entropy
        values
        """

        # ======================================================
        # Differentiable communication reconstruction
        # ======================================================

        if communication_source_obs is not None:
            received_messages, reconstructed_trust = (
                self._reconstruct_received_messages(
                    communication_source_obs=communication_source_obs,
                    communication_field_ids=communication_field_ids,
                    agent_ids=agent_ids,
                    communication_valid=communication_valid,
                    trust_weights=trust_weights,
                )
            )

            if reconstructed_trust is not None:

                trust_weights = (
                    reconstructed_trust
                )

        # ======================================================
        # Actor
        # ======================================================

        communication_log_probs = None
        communication_entropy = None
        if (
            communication_source_obs is not None
            and communication_field_ids is not None
        ):

            batch_size = communication_source_obs.shape[0]

            sender_obs = communication_source_obs.reshape(
                batch_size * NUM_AGENTS,
                OBS_DIM,
            )

            sender_hidden = self.actor.get_local_hidden(
                sender_obs
            )

            # Select the receiver's relevant sender observations.
            #
            # We need all NUM_AGENTS because the decoder produces
            # a message for every sender.
            sender_hidden = sender_hidden.reshape(
                batch_size,
                NUM_AGENTS,
                -1,
            )

            # Evaluate stored message IDs under CURRENT decoder.
            #
            # evaluate_message() expects [B, hidden] so flatten first.
            flat_sender_hidden = sender_hidden.reshape(
                batch_size * NUM_AGENTS,
                -1,
            )

            flat_field_ids = {
                key: value.reshape(
                    batch_size * NUM_AGENTS
                )
                for key, value in {
                    "event_type": communication_field_ids[:, :, 0],
                    "target_type": communication_field_ids[:, :, 1],
                    "threat_level": communication_field_ids[:, :, 2],
                    "status": communication_field_ids[:, :, 3],
                    "priority": communication_field_ids[:, :, 4],
                    "confidence": communication_field_ids[:, :, 5],
                    "target_id": communication_field_ids[:, :, 6],
                }.items()
            }

            message_log_probs, message_entropies = (
                self.communication.decoder.evaluate_message(
                    flat_sender_hidden,
                    flat_field_ids,
                )
            )

            communication_log_probs = torch.stack(
                list(message_log_probs.values()),
                dim=-1,
            ).sum(dim=-1)

            communication_entropy = torch.stack(
                list(message_entropies.values()),
                dim=-1,
            ).mean(dim=-1)

            communication_log_probs = (
                communication_log_probs.reshape(
                    batch_size,
                    NUM_AGENTS,
                )
            )

            communication_entropy = (
                communication_entropy.reshape(
                    batch_size,
                    NUM_AGENTS,
                )
            )

        logits = self.actor_forward(
            observations,
            action_masks,
            received_messages=received_messages,
            trust_weights=trust_weights,
        )

        # ======================================================
        # Action distribution
        # ======================================================

        distribution = Categorical(
            logits=logits
        )

        log_probs = distribution.log_prob(
            actions
        )

        entropy = distribution.entropy()

        # ======================================================
        # Centralized critic
        # ======================================================

        values = self.critic_forward(
            global_states
        )

        values = values[
            torch.arange(
                values.size(0),
                device=self.device,
            ),
            agent_ids,
        ]

        # return (
        #     log_probs,
        #     entropy,
        #     values,
        # )
        return (
            log_probs,
            entropy,
            values,
            communication_log_probs,
            communication_entropy,
        )

    # ==========================================================
    # Save
    # ==========================================================

    def save(
        self,
        path,
    ):
        """
        Save MAPPO checkpoint.
        """

        trust_state = None

        if hasattr(
            self.communication,
            "get_trust_state",
        ):

            trust_state = (
                self.communication.get_trust_state()
            )

        checkpoint = {

            "model":
                self.model.state_dict(),

            "actor_optimizer":
                self.actor_optimizer.state_dict(),

            "critic_optimizer":
                self.critic_optimizer.state_dict(),

            "value_norm":
                None
                if self.value_norm is None
                else self.value_norm.state_dict(),

            "trust_state":
                trust_state,
        }

        torch.save(
            checkpoint,
            path,
        )

    # ==========================================================
    # Load
    # ==========================================================

    def load(
        self,
        path,
    ):
        """
        Load MAPPO checkpoint.

        Old checkpoints without trust state remain compatible.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        if (
            self.value_norm is not None
            and checkpoint.get(
                "value_norm"
            ) is not None
        ):

            self.value_norm.load_state_dict(
                checkpoint["value_norm"]
            )

        trust_state = checkpoint.get(
            "trust_state"
        )

        if (
            trust_state is not None
            and hasattr(
                self.communication,
                "load_trust_state",
            )
        ):

            self.communication.load_trust_state(
                trust_state
            )

    # ==========================================================
    # PPO UPDATE
    # ==========================================================

    def update(
        self,
        buffer,
    ):
        """
        Perform one MAPPO update.

        Communication training
        ----------------------

        The buffer provides:

            communication_source_obs
            communication_valid
            trust_weights

        The source observations are passed through the CURRENT
        sender communication network during every PPO minibatch.

        Therefore the PPO gradient can reach:

            receiver policy
                ^
                |
        communication attention
                ^
                |
             encoder
                ^
                |
             decoder
                ^
                |
          sender actor

        ``received_messages`` is deliberately NOT used as the
        differentiable source when ``communication_source_obs``
        exists.
        """

        batch = buffer.get_batches()

        # ======================================================
        # Core PPO tensors
        # ======================================================

        obs = batch["obs"]

        global_obs = batch["global_obs"]

        actions = batch["actions"]

        old_log_probs = batch["log_probs"]

        returns = batch["returns"]

        advantages = batch["advantages"]

        action_masks = batch["action_masks"]

        # ======================================================
        # Communication tensors
        # ======================================================

        communication_source_obs = batch.get(
            "communication_source_obs",
            None,
        )

        communication_valid = batch.get(
            "communication_valid",
            None,
        )

        trust_weights = batch.get(
            "trust_weights",
            None,
        )
        communication_field_ids = batch.get(
            "communication_field_ids",
            None,
        )

        old_communication_log_probs = batch.get(
            "communication_log_probs",
            None,
        )

        # ------------------------------------------------------
        # Fallback for older buffers
        # ------------------------------------------------------

        received_messages = batch.get(
            "received_messages",
            None,
        )

        # ======================================================
        # Dimensions
        # ======================================================

        T = obs.shape[0]

        # ======================================================
        # Flatten local actor data
        # ======================================================

        obs = obs.reshape(
            T * NUM_AGENTS,
            OBS_DIM,
        )

        actions = actions.reshape(
            T * NUM_AGENTS,
        )

        old_log_probs = old_log_probs.reshape(
            T * NUM_AGENTS,
        )

        returns = returns.reshape(
            T * NUM_AGENTS,
        )

        advantages = advantages.reshape(
            T * NUM_AGENTS,
        )
        communication_field_ids = (
            communication_field_ids
            if communication_field_ids is None
            else communication_field_ids
        )
        old_communication_log_probs = (
            old_communication_log_probs
            if old_communication_log_probs is None
            else old_communication_log_probs
        )

        action_masks = action_masks.reshape(
            T * NUM_AGENTS,
            -1,
        )
        # commented , now dont point out this
        # if communication_source_obs is not None:
        #     communication_source_obs = (
        #         communication_source_obs.reshape(
        #             T * NUM_AGENTS,
        #             OBS_DIM,
        #         )
        #     )

        # ======================================================
        # Flatten communication source observations
        # ======================================================

        if communication_source_obs is not None:

            communication_source_obs = (
                communication_source_obs.to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )

            if communication_source_obs.shape != (
                T,
                NUM_AGENTS,
                NUM_AGENTS,
                OBS_DIM,
            ):

                # Current buffer stores:
                #
                # [T, N, OBS]
                #
                # where each timestep contains the previous
                # timestep's complete multi-agent observation.
                #
                # Expand receiver dimension later.
                if communication_source_obs.shape == (
                    T,
                    NUM_AGENTS,
                    OBS_DIM,
                ):

                    pass

                else:

                    raise ValueError(
                        "Unexpected communication_source_obs "
                        "shape: "
                        f"{tuple(communication_source_obs.shape)}"
                    )

        # ======================================================
        # Flatten communication validity
        # ======================================================

        if communication_valid is not None:

            communication_valid = (
                communication_valid.to(
                    device=self.device,
                    dtype=torch.bool,
                )
            )

        # ======================================================
        # Flatten fallback messages
        # ======================================================

        if received_messages is not None:

            received_messages = (
                received_messages.reshape(
                    T * NUM_AGENTS,
                    NUM_AGENTS,
                    COMMUNICATION_DIM,
                )
            )

        # ======================================================
        # Trust
        #
        # Buffer convention:
        #
        # trust_weights[t, receiver, sender]
        #
        # We need:
        #
        # [T*N, sender]
        #
        # so each receiver sample gets its own trust vector.
        # ======================================================

        if trust_weights is not None:

            trust_weights = (
                trust_weights.reshape(
                    T * NUM_AGENTS,
                    NUM_AGENTS,
                )
            )

        # ======================================================
        # Centralized critic input
        # ======================================================

        global_obs = (
            global_obs.repeat_interleave(
                NUM_AGENTS,
                dim=0,
            )
        )

        # ======================================================
        # Agent IDs
        #
        # Pattern:
        #
        # timestep 0:
        #   0 1 2 3 4
        #
        # timestep 1:
        #   0 1 2 3 4
        #
        # ...
        # ======================================================

        agent_ids = torch.arange(
            NUM_AGENTS,
            device=self.device,
        ).repeat(T)

        # ======================================================
        # Advantage normalization
        # ======================================================

        if NORMALIZE_ADVANTAGES:

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std()
                + 1e-8
            )

        # ======================================================
        # Value normalization
        # ======================================================

        if self.value_norm is not None:

            self.value_norm.update(
                returns
            )

        # ======================================================
        # Statistics
        # ======================================================

        actor_loss_epoch = 0.0

        critic_loss_epoch = 0.0

        entropy_epoch = 0.0

        # ======================================================
        # Dataset size
        # ======================================================

        dataset_size = obs.shape[0]

        # ======================================================
        # PPO epochs
        # ======================================================

        for epoch in range(
            UPDATE_EPOCHS
        ):

            permutation = torch.randperm(
                dataset_size,
                device=self.device,
            )

            # --------------------------------------------------
            # Minibatches
            # --------------------------------------------------

            for start in range(
                0,
                dataset_size,
                MINIBATCH_SIZE,
            ):

                end = (
                    start
                    + MINIBATCH_SIZE
                )

                idx = permutation[
                    start:end
                ]

                # ==================================================
                # Core minibatch
                # ==================================================

                mb_obs = obs[idx]

                mb_global = (
                    global_obs[idx]
                )

                mb_actions = (
                    actions[idx]
                )

                mb_old_log_probs = (
                    old_log_probs[idx]
                )

                mb_returns = (
                    returns[idx]
                )

                mb_advantages = (
                    advantages[idx]
                )

                mb_action_masks = (
                    action_masks[idx]
                )

                mb_agent_ids = (
                    agent_ids[idx]
                )
                mb_comm_field_ids = None
                mb_old_comm_log_probs = None
                if communication_field_ids is not None:
                    comm_ids = (
                        communication_field_ids
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                            7,
                        )
                    )
                    mb_comm_field_ids = comm_ids[idx]
                if old_communication_log_probs is not None:
                    # Buffer stores log-probability for each of the
                    # 7 message fields:
                    #
                    # [T, N, 7]
                    #
                    # PPO needs the joint message log-prob:
                    #
                    # [T, N]
                    #
                    # log P(message) = sum of field log-probs

                    old_comm = (
                        old_communication_log_probs.sum(dim=-1)
                    )

                    # [T, N]
                    #
                    # Expand receiver dimension so every receiver
                    # gets the sender log-probabilities.

                    old_comm = (
                        old_comm
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                        )
                    )

                    mb_old_comm_log_probs = old_comm[idx]

                # ==================================================
                # Value targets
                # ==================================================

                if self.value_norm is not None:

                    normalized_returns = (
                        self.value_norm.normalize(
                            mb_returns
                        )
                    )

                else:

                    normalized_returns = (
                        mb_returns
                    )

                # ==================================================
                # COMMUNICATION SOURCE
                # ==================================================

                mb_source_obs = None
                mb_comm_valid = None

                if communication_source_obs is not None:

                    # ------------------------------------------------------
                    # Buffer:
                    #
                    # communication_source_obs
                    #     [T, N, OBS]
                    #
                    # Expand the receiver dimension:
                    #
                    #     [T, receiver, sender, OBS]
                    #
                    # Then flatten timestep + receiver:
                    #
                    #     [T*N, sender, OBS]
                    #
                    # This now has the SAME ordering as:
                    #
                    # obs
                    # actions
                    # agent_ids
                    # ------------------------------------------------------

                    source = (
                        communication_source_obs
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                            OBS_DIM,
                        )
                    )

                    mb_source_obs = (
                        source[idx]
                    )

                    # ------------------------------------------------------
                    # Communication validity
                    # ------------------------------------------------------

                    if communication_valid is not None:

                        valid = (
                            communication_valid
                            .unsqueeze(1)
                            .expand(
                                -1,
                                NUM_AGENTS,
                            )
                            .reshape(
                                T * NUM_AGENTS
                            )
                        )

                        mb_comm_valid = (
                            valid[idx]
                        )

                # ==================================================
                # Fallback communication
                # ==================================================

                mb_received_messages = None

                if (
                    mb_source_obs is None
                    and received_messages is not None
                ):

                    mb_received_messages = (
                        received_messages[idx]
                    )

                # ==================================================
                # Trust
                # ==================================================

                mb_trust_weights = None

                if trust_weights is not None:

                    mb_trust_weights = (
                        trust_weights[idx]
                    )

                # ==================================================
                # Forward
                # ==================================================

                (
                    new_log_probs,
                    entropy,
                    values,
                    new_communication_log_probs,
                    communication_entropy,
                ) = self.evaluate_actions(
                    mb_obs,
                    mb_global,
                    mb_actions,
                    mb_agent_ids,
                    mb_action_masks,
                    received_messages=(
                        mb_received_messages
                    ),
                    communication_field_ids=mb_comm_field_ids,
                    trust_weights=(
                        mb_trust_weights
                    ),

                    communication_source_obs=(
                        mb_source_obs
                    ),

                    communication_valid=(
                        mb_comm_valid
                    ),
                )

                # ==================================================
                # PPO ratio
                # ==================================================

                ratio = torch.exp(
                    new_log_probs
                    - mb_old_log_probs
                )
                communication_ratio = torch.exp(
                    new_communication_log_probs
                    - mb_old_comm_log_probs
                )
                comm_surrogate1 = (
                    communication_ratio
                    * mb_advantages.view(-1, 1)
                )

                comm_surrogate2 = (
                    torch.clamp(
                        communication_ratio,
                        1.0 - PPO_CLIP,
                        1.0 + PPO_CLIP,
                    )
                    * mb_advantages.view(-1, 1)
                )

                communication_actor_loss = (
                    -torch.min(
                        comm_surrogate1,
                        comm_surrogate2,
                    ).mean()
                )
                # ==================================================
                # Clipped objective
                # ==================================================

                surrogate1 = (
                    ratio
                    * mb_advantages
                )

                surrogate2 = (
                    torch.clamp(
                        ratio,
                        1.0 - PPO_CLIP,
                        1.0 + PPO_CLIP,
                    )
                    * mb_advantages
                )

                # ==================================================
                # Actor loss
                # ==================================================

                actor_loss = (
                    -torch.min(
                        surrogate1,
                        surrogate2,
                    ).mean()
                )

                # ==================================================
                # Critic loss
                # ==================================================

                critic_loss = F.mse_loss(
                    values,
                    normalized_returns,
                )

                # ==================================================
                # Entropy
                # ==================================================

                entropy_loss = (
                    entropy.mean() + communication_entropy.mean()
                )

                # ==================================================
                # Statistics
                # ==================================================

                actor_loss_epoch += (
                    actor_loss.item()
                )

                critic_loss_epoch += (
                    critic_loss.item()
                )

                entropy_epoch += (
                    entropy_loss.item()
                )

                # ==================================================
                # Total loss
                # ==================================================

                total_loss = (
                    actor_loss
                    + communication_actor_loss
                    + VALUE_LOSS_COEF * critic_loss
                    - ENTROPY_COEF * entropy_loss
                )

                # ==================================================
                # Zero gradients
                # ==================================================

                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )

                self.critic_optimizer.zero_grad(
                    set_to_none=True
                )

                # ==================================================
                # Backpropagation
                # ==================================================

                total_loss.backward()

                # # temp :-
                # decoder_grad = 0.0
                # encoder_grad = 0.0

                # for name, param in self.communication.decoder.named_parameters():
                #     if param.grad is not None:
                #         decoder_grad += param.grad.abs().sum().item()

                # for name, param in self.communication.encoder.named_parameters():
                #     if param.grad is not None:
                #         encoder_grad += param.grad.abs().sum().item()

                # print(
                #     f"[COMM GRAD] decoder={decoder_grad:.6e} "
                #     f"encoder={encoder_grad:.6e}"
                # )

                # ==================================================
                # Gradient clipping
                # ==================================================

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )

                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                )

                # ==================================================
                # Optimizer step
                # ==================================================

                self.actor_optimizer.step()

                self.critic_optimizer.step()

        # ======================================================
        # Average statistics
        # ======================================================

        num_updates = (
            UPDATE_EPOCHS
            * (
                (
                    dataset_size
                    + MINIBATCH_SIZE
                    - 1
                )
                // MINIBATCH_SIZE
            )
        )

        training_stats = {

            "actor_loss":
                actor_loss_epoch
                / num_updates,

            "critic_loss":
                critic_loss_epoch
                / num_updates,

            "entropy":
                entropy_epoch
                / num_updates,
        }

        return training_stats
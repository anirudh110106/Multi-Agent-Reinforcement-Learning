"""
gnn_attention.py

Neural networks for MAPPO with hierarchical GNN + attention and
structured communication.

Architecture
------------

Shared Actor:

    Local Observation
            |
            v
    Split into entities:
        mission            (1 node)
        subnet context     (NUM_HQ_SUBNETS nodes)
        per-host alerts     (NUM_HQ_SUBNETS * MAX_HOSTS nodes)
            |
            v
    Hierarchical Graph Message Passing
        host      <-> own subnet
        host      <-> other hosts, same subnet
        subnet    <-> subnet            (fully connected unless
                                          SUBNET_EDGES supplied)
        subnet    <-> mission
            |
            v
    Multi-Head Attention (over ALL graph nodes)
            |
            v
    Pool back down to [mission, subnet_0 .. subnet_N] tokens
    (host-level detail has already been propagated upward by
    the GNN, so we don't need to carry raw host tokens forward)
            |
            v
        Local Hidden
        /            \
       /              \
      v                v
 Policy Path      Communication Path
                        |
                        v
                    MessageDecoder
                        |
                        v
                 Structured Message
                        |
                        v
                    MessageEncoder
                        |
                        v
               Communication Vector
                        |
                        v
                Other Blue Agents
                        |
                        v
              Trust-Weighted Communication
                        |
                        v
              Communication Attention
                        |
                        v
               Action Representation
                        |
                        v
                   Action Logits


Central Critic (unchanged):

    Global Observation
            |
            v
      Per-Agent Tokens
            |
            v
    Cross-Agent Attention
            |
            v
       Value per Agent


WHY THIS IS DIFFERENT FROM A "GNN" OVER MISSION+SUBNET NODES ONLY
-------------------------------------------------------------------
A GNN over ~4-10 fully-connected mission/subnet nodes with a fixed,
uniform adjacency is functionally close to a no-op sitting in front
of MultiheadAttention: attention already learns adaptive pairwise
weights over that same fully-connected node set, so a fixed uniform
message-passing layer on top of it adds parameters without adding
real inductive bias. That's the most likely reason a GNN+attention
version built that way doesn't beat plain ANN+attention.

The observation already contains host-level signal
(`process_alerts`, `connection_alerts`, each length MAX_HOSTS,
packed into every subnet block) that the old code flattened into an
opaque per-subnet vector. This version promotes that host-level
signal into real graph nodes with a real (non-fully-connected)
topology: each host only connects to its own subnet and to other
hosts in that same subnet, and subnets connect to mission. That
gives the GNN actual structure to propagate over before attention
re-summarizes it -- the two modules are now doing different jobs
instead of the same job twice.

If you know the real CC4 subnet-to-subnet connectivity for this
agent's zone, pass it in via `subnet_edges` in SharedActor.__init__
(list of (i, j) index pairs into the NUM_HQ_SUBNETS subnets) instead
of relying on the fully-connected fallback -- that is the single
highest-value upgrade left on the table here, since inter-subnet
adjacency is fixed and known at training time even though it isn't
exposed in the flat observation itself.

IMPORTANT
---------
The external interface is intentionally unchanged.

The model still accepts:

    observation
    received_messages
    trust_weights

and still returns the same outputs, with the same shapes.

No changes are required in:

    mappo.py
    buffer.py
    train.py
    env.py
    encoder.py
    decoder.py
    schema.py
    structured_communication.py

`local_projection`'s input dimension is still
NUM_ENTITY_TOKENS * EMBED_DIM where NUM_ENTITY_TOKENS = 1 +
NUM_HQ_SUBNETS, exactly as in the pre-GNN version, so
COMMUNICATION_LATENT_DIM and everything downstream of it (decoder,
encoder, trust, evaluator) is untouched.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    OBS_DIM,
    ACTION_DIM,
    EMBED_DIM,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    NUM_AGENTS,
)

from CybORG.Agents.Wrappers.BlueFlatWrapper import (
    NUM_SUBNETS,
    NUM_HQ_SUBNETS,
    MAX_HOSTS,
)

from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (
    NUM_MESSAGES,
    MESSAGE_LENGTH,
)

from .communication.structured_communication import (
    StructuredCommunication,
)


# ==========================================================
# Communication Configuration
# ==========================================================

COMMUNICATION_DIM = 128

COMMUNICATION_LATENT_DIM = 256

COMMUNICATION_ATTENTION_DIM = EMBED_DIM


# ==========================================================
# Entity layout
#
# Each subnet block (SUBNET_BLOCK_DIM) is, in order:
#
#   [ subnet one-hot          | NUM_SUBNETS ]
#   [ blocked-subnets mask    | NUM_SUBNETS ]
#   [ communication-policy    | NUM_SUBNETS ]
#   [ process alerts          | MAX_HOSTS   ]
#   [ connection alerts       | MAX_HOSTS   ]
#
# The first three chunks are context shared by the whole subnet.
# The last two chunks are genuinely per-host and are what we turn
# into host graph nodes below.
# ==========================================================

MISSION_DIM = 1

SUBNET_CONTEXT_DIM = 3 * NUM_SUBNETS

HOST_FEATURE_DIM = 2  # [process_alert_i, connection_alert_i]

SUBNET_BLOCK_DIM = (
    SUBNET_CONTEXT_DIM
    + 2 * MAX_HOSTS
)

MESSAGE_DIM = NUM_MESSAGES * MESSAGE_LENGTH

# Tokens kept AFTER pooling, fed into local_projection.
# Identical to the pre-GNN version -- this is what keeps every
# downstream module's input shape unchanged.
NUM_ENTITY_TOKENS = (
    1
    + NUM_HQ_SUBNETS
)

# Full graph size used DURING message passing / attention, before
# pooling back down to NUM_ENTITY_TOKENS.
NUM_HOST_NODES = NUM_HQ_SUBNETS * MAX_HOSTS

NUM_GRAPH_NODES = (
    NUM_ENTITY_TOKENS
    + NUM_HOST_NODES
)


_EXPECTED_OBS_DIM = (
    MISSION_DIM
    + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM
    + MESSAGE_DIM
)

assert _EXPECTED_OBS_DIM == OBS_DIM, (
    f"Entity split ({_EXPECTED_OBS_DIM}) does not match "
    f"OBS_DIM ({OBS_DIM}) -- "
    "BlueFlatWrapper's per-subnet block size probably "
    "doesn't match the configured observation dimensions."
)


# Optional: real subnet-to-subnet adjacency for this agent's zone,
# as (i, j) index pairs into range(NUM_HQ_SUBNETS). Leave as None to
# fall back to a fully-connected subnet graph.
SUBNET_EDGES: Optional[List[Tuple[int, int]]] = None


# ==========================================================
# Utility
# ==========================================================

def build_mlp(
    input_dim: int,
    output_dim: int,
):
    """
    Build the standard MAPPO MLP.
    """

    layers = []

    current = input_dim

    for _ in range(NUM_HIDDEN_LAYERS):

        layers.append(nn.Linear(current, HIDDEN_DIM))
        layers.append(nn.ReLU())

        current = HIDDEN_DIM

    layers.append(nn.Linear(current, output_dim))

    return nn.Sequential(*layers)


# ==========================================================
# Hierarchical adjacency
# ==========================================================

def _build_hierarchical_adjacency(
    num_subnets: int,
    max_hosts: int,
    subnet_edges: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """
    Build a fixed, row-normalized adjacency matrix over:

        node 0                        = mission
        node 1 .. num_subnets         = subnets
        node (1+num_subnets) ..       = hosts, grouped by subnet

    Edges:

        mission <-> every subnet
        subnet  <-> subnet             (subnet_edges, or fully
                                         connected if not given)
        subnet  <-> its own hosts
        host    <-> other hosts in the SAME subnet only

    Hosts do NOT connect across subnets and do NOT connect
    directly to mission -- that's the actual topological prior
    this graph encodes, as opposed to a flat fully-connected blob.
    """

    num_nodes = 1 + num_subnets + num_subnets * max_hosts

    adjacency = torch.zeros(num_nodes, num_nodes)

    mission_idx = 0
    subnet_start = 1
    host_start = 1 + num_subnets

    def host_idx(subnet_i: int, host_j: int) -> int:
        return host_start + subnet_i * max_hosts + host_j

    # mission <-> subnets
    for s in range(num_subnets):
        si = subnet_start + s
        adjacency[mission_idx, si] = 1.0
        adjacency[si, mission_idx] = 1.0

    # subnet <-> subnet
    if subnet_edges is not None:
        for (i, j) in subnet_edges:
            si, sj = subnet_start + i, subnet_start + j
            adjacency[si, sj] = 1.0
            adjacency[sj, si] = 1.0
    else:
        for s1 in range(num_subnets):
            for s2 in range(num_subnets):
                if s1 != s2:
                    si1 = subnet_start + s1
                    si2 = subnet_start + s2
                    adjacency[si1, si2] = 1.0

    # subnet <-> own hosts, host <-> host (same subnet)
    for s in range(num_subnets):
        si = subnet_start + s
        for h in range(max_hosts):
            hi = host_idx(s, h)
            adjacency[si, hi] = 1.0
            adjacency[hi, si] = 1.0
            for h2 in range(max_hosts):
                if h2 != h:
                    hi2 = host_idx(s, h2)
                    adjacency[hi, hi2] = 1.0

    # mean aggregation
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    adjacency = adjacency / degree

    return adjacency


# ==========================================================
# Graph Neural Network Layer
# ==========================================================

class GraphMessagePassing(nn.Module):
    """
    Gated graph message-passing layer over a fixed adjacency.

    Input:  [B, N, D]
    Output: [B, N, D]
    """

    def __init__(
        self,
        dim: int,
        adjacency: torch.Tensor,
    ):
        super().__init__()

        self.dim = dim

        self.self_linear = nn.Linear(dim, dim)
        self.neighbor_linear = nn.Linear(dim, dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(dim)

        self.register_buffer("adjacency", adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        neighbor_info = torch.matmul(self.adjacency, x)

        self_info = self.self_linear(x)
        neighbor_info = self.neighbor_linear(neighbor_info)

        gate_input = torch.cat([self_info, neighbor_info], dim=-1)
        gate = self.gate(gate_input)

        updated = self_info + gate * neighbor_info
        updated = self.norm(x + updated)

        return updated


# ==========================================================
# Shared Actor
# ==========================================================

class SharedActor(nn.Module):
    """
    One policy shared by ALL Blue agents.
    """

    def __init__(
        self,
        num_agents: int = NUM_AGENTS,
        communication_dim: int = COMMUNICATION_DIM,
        communication_latent_dim: int = COMMUNICATION_LATENT_DIM,
        num_targets: Optional[int] = None,
        use_host_graph: bool = True,
        subnet_edges: Optional[List[Tuple[int, int]]] = SUBNET_EDGES,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.communication_dim = communication_dim
        self.communication_latent_dim = communication_latent_dim
        self.use_host_graph = use_host_graph and MAX_HOSTS > 0

        # ------------------------------------------------------
        # Local observation embeddings
        # ------------------------------------------------------

        self.mission_embed = nn.Linear(MISSION_DIM, EMBED_DIM)
        self.subnet_embed = nn.Linear(SUBNET_CONTEXT_DIM, EMBED_DIM)

        if self.use_host_graph:
            self.host_embed = nn.Linear(HOST_FEATURE_DIM, EMBED_DIM)
            num_graph_nodes = NUM_GRAPH_NODES
        else:
            self.host_embed = None
            num_graph_nodes = NUM_ENTITY_TOKENS

        self.num_graph_nodes = num_graph_nodes

        # ------------------------------------------------------
        # Positional embeddings (over the FULL graph, pre-pooling)
        # ------------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_graph_nodes, EMBED_DIM)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # ------------------------------------------------------
        # GNN
        # ------------------------------------------------------

        if self.use_host_graph:
            adjacency = _build_hierarchical_adjacency(
                num_subnets=NUM_HQ_SUBNETS,
                max_hosts=MAX_HOSTS,
                subnet_edges=subnet_edges,
            )
            self.gnn1 = GraphMessagePassing(EMBED_DIM, adjacency.clone())
            self.gnn2 = GraphMessagePassing(EMBED_DIM, adjacency.clone())
        else:
            self.gnn1 = None
            self.gnn2 = None

        # ------------------------------------------------------
        # Multi-Head Attention (over the full graph)
        # ------------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        # ------------------------------------------------------
        # Local representation
        #
        # Pooled back down to NUM_ENTITY_TOKENS (mission + subnets)
        # -- same shape as the pre-GNN version.
        # ------------------------------------------------------

        self.local_projection = nn.Sequential(
            nn.Linear(
                NUM_ENTITY_TOKENS * EMBED_DIM,
                communication_latent_dim,
            ),
            nn.LayerNorm(communication_latent_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------
        # Structured communication module (unchanged)
        # ------------------------------------------------------

        self.communication = StructuredCommunication(
            input_dim=communication_latent_dim,
            message_dim=communication_dim,
            num_agents=num_agents,
            num_targets=num_targets,
        )

        # ------------------------------------------------------
        # Communication attention (unchanged)
        # ------------------------------------------------------

        self.communication_query = nn.Linear(
            communication_latent_dim,
            COMMUNICATION_ATTENTION_DIM,
        )

        self.communication_key = nn.Linear(
            communication_dim,
            COMMUNICATION_ATTENTION_DIM,
        )

        self.communication_value = nn.Linear(
            communication_dim,
            COMMUNICATION_ATTENTION_DIM,
        )

        self.communication_attention = nn.MultiheadAttention(
            embed_dim=COMMUNICATION_ATTENTION_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.communication_norm = nn.LayerNorm(
            COMMUNICATION_ATTENTION_DIM
        )

        # ------------------------------------------------------
        # Policy fusion (unchanged)
        # ------------------------------------------------------

        self.policy_input_projection = nn.Sequential(
            nn.Linear(
                communication_latent_dim + COMMUNICATION_ATTENTION_DIM,
                communication_latent_dim,
            ),
            nn.LayerNorm(communication_latent_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------
        # Policy head (unchanged)
        # ------------------------------------------------------

        self.policy_head = build_mlp(
            communication_latent_dim,
            ACTION_DIM,
        )

        # ------------------------------------------------------
        # Debug / visualization state
        # ------------------------------------------------------

        self.last_attention = None
        self.last_communication_attention = None
        self.last_outgoing_message = None
        self.last_local_hidden = None
        self.last_gnn_output = None

    # ======================================================
    # Observation splitting
    # ======================================================

    def _split_entities(self, observation: torch.Tensor):
        """
        Split [mission | subnets | native messages] from the
        flat CC4 observation. Unchanged from the pre-GNN version.
        """

        subnets_start = MISSION_DIM
        subnets_end = (
            MISSION_DIM
            + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM
        )

        mission = observation[:, :MISSION_DIM]

        subnets = observation[
            :, subnets_start:subnets_end
        ].view(-1, NUM_HQ_SUBNETS, SUBNET_BLOCK_DIM)

        messages = observation[
            :, subnets_end:subnets_end + MESSAGE_DIM
        ]

        return mission, subnets, messages

    # ======================================================
    # Entity encoding
    # ======================================================

    def _encode_entities(self, observation: torch.Tensor):
        """
        Flat observation -> hierarchical graph -> GNN -> attention
        -> pooled entity tokens (mission + subnets).
        """

        mission, subnets, _messages = self._split_entities(
            observation
        )

        batch_size = observation.shape[0]

        # --------------------------------------------------
        # Mission node
        # --------------------------------------------------

        mission_tok = torch.relu(
            self.mission_embed(mission)
        ).unsqueeze(1)

        # --------------------------------------------------
        # Subnet context nodes (one-hot + blocked + comms mask
        # only -- host alerts are split off separately below)
        # --------------------------------------------------

        subnet_context = subnets[..., :SUBNET_CONTEXT_DIM]

        subnet_tok = torch.relu(
            self.subnet_embed(subnet_context)
        )

        if not self.use_host_graph:
            tokens = torch.cat([mission_tok, subnet_tok], dim=1)
            tokens = tokens + self.pos_embed

            attn_out, attention_weights = self.attention(
                tokens, tokens, tokens
            )
            self.last_attention = attention_weights.detach()

            x = self.norm1(tokens + attn_out)
            x = self.norm2(x + self.ffn(x))

            return x

        # --------------------------------------------------
        # Host nodes: real per-host alert signal, previously
        # flattened into an opaque per-subnet vector.
        # --------------------------------------------------

        process_alerts = subnets[
            ..., SUBNET_CONTEXT_DIM:SUBNET_CONTEXT_DIM + MAX_HOSTS
        ]
        connection_alerts = subnets[
            ...,
            SUBNET_CONTEXT_DIM + MAX_HOSTS:
            SUBNET_CONTEXT_DIM + 2 * MAX_HOSTS,
        ]

        host_features = torch.stack(
            [process_alerts, connection_alerts], dim=-1
        )  # [B, NUM_HQ_SUBNETS, MAX_HOSTS, 2]

        host_features = host_features.reshape(
            batch_size, NUM_HOST_NODES, HOST_FEATURE_DIM
        )

        host_tok = torch.relu(
            self.host_embed(host_features)
        )

        # --------------------------------------------------
        # Full graph: [mission | subnets | hosts]
        # --------------------------------------------------

        tokens = torch.cat(
            [mission_tok, subnet_tok, host_tok], dim=1
        )
        tokens = tokens + self.pos_embed

        # --------------------------------------------------
        # Hierarchical GNN message passing
        # --------------------------------------------------

        tokens = self.gnn1(tokens)
        tokens = F.gelu(tokens)

        tokens = self.gnn2(tokens)
        tokens = F.gelu(tokens)

        self.last_gnn_output = tokens.detach()

        # --------------------------------------------------
        # Attention over the full graph
        # --------------------------------------------------

        attn_out, attention_weights = self.attention(
            tokens, tokens, tokens
        )
        self.last_attention = attention_weights.detach()

        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ffn(x))

        # --------------------------------------------------
        # Pool back down to mission + subnet tokens. Host-level
        # detail has already been propagated into these via the
        # GNN + attention above, so we don't carry raw host
        # tokens into local_projection.
        # --------------------------------------------------

        entity_tokens = x[:, :NUM_ENTITY_TOKENS, :]

        return entity_tokens

    # ======================================================
    # Local hidden representation
    # ======================================================

    def _get_local_hidden(self, observation: torch.Tensor):

        x = self._encode_entities(observation)

        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)

        local_hidden = self.local_projection(flat)

        self.last_local_hidden = local_hidden.detach()

        return local_hidden

    # ======================================================
    # Communication generation (unchanged)
    # ======================================================

    def generate_communication(self, local_hidden: torch.Tensor):

        (
            field_ids,
            log_probs,
            entropies,
            communication_vector,
        ) = self.communication.generate_message(local_hidden)

        self.last_outgoing_message = communication_vector.detach()

        return field_ids, log_probs, entropies, communication_vector

    # ======================================================
    # Received communication (unchanged)
    # ======================================================

    def _prepare_received_messages(
        self,
        received_messages: Optional[torch.Tensor],
        trust_weights: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):

        if received_messages is None:
            return None, None

        messages = received_messages

        if messages.dim() == 2:
            messages = messages.unsqueeze(0)

        if messages.dim() != 3:
            raise ValueError(
                "received_messages must have shape "
                "[B, N, D] or [B, N-1, D]."
            )

        messages = messages.to(device=device, dtype=dtype)

        if trust_weights is not None:

            if trust_weights.dim() == 1:
                trust_weights = trust_weights.unsqueeze(0)

            trust_weights = trust_weights.to(device=device, dtype=dtype)

            if trust_weights.dim() != 2:
                raise ValueError(
                    "trust_weights must have shape "
                    "[B, N] or [B, N-1]."
                )

            if trust_weights.shape[0] == 1 and batch_size > 1:
                trust_weights = trust_weights.expand(batch_size, -1)

        return messages, trust_weights

    # ======================================================
    # Communication attention (unchanged)
    # ======================================================

    def _apply_received_communication(
        self,
        local_hidden: torch.Tensor,
        received_messages: Optional[torch.Tensor],
        trust_weights: Optional[torch.Tensor] = None,
    ):

        batch_size = local_hidden.shape[0]
        device = local_hidden.device
        dtype = local_hidden.dtype

        if received_messages is None:
            return torch.zeros(
                batch_size,
                COMMUNICATION_ATTENTION_DIM,
                device=device,
                dtype=dtype,
            )

        messages, trust = self._prepare_received_messages(
            received_messages, trust_weights, batch_size, device, dtype
        )

        keys = self.communication_key(messages)
        values = self.communication_value(messages)

        if trust is not None:

            if trust.shape[1] != messages.shape[1]:
                raise ValueError(
                    "trust_weights and received_messages must "
                    "contain the same number of senders."
                )

            trust_safe = torch.clamp(trust, min=1e-4, max=1.0)
            values = values * trust_safe.unsqueeze(-1)

        query = self.communication_query(local_hidden).unsqueeze(1)

        context, communication_weights = self.communication_attention(
            query, keys, values, need_weights=True
        )

        self.last_communication_attention = communication_weights.detach()

        context = context.squeeze(1)
        context = self.communication_norm(context)

        return context

    # ======================================================
    # Main forward (unchanged interface)
    # ======================================================

    def forward(
        self,
        observation: torch.Tensor,
        received_messages: Optional[torch.Tensor] = None,
        trust_weights: Optional[torch.Tensor] = None,
        return_communication: bool = False,
    ):

        squeeze_output = observation.dim() == 1

        if squeeze_output:
            observation = observation.unsqueeze(0)

        local_hidden = self._get_local_hidden(observation)

        (
            field_ids,
            message_log_probs,
            message_entropies,
            outgoing_message,
        ) = self.generate_communication(local_hidden)

        communication_context = self._apply_received_communication(
            local_hidden=local_hidden,
            received_messages=received_messages,
            trust_weights=trust_weights,
        )

        policy_representation = torch.cat(
            [local_hidden, communication_context], dim=-1
        )
        policy_representation = self.policy_input_projection(
            policy_representation
        )

        logits = self.policy_head(policy_representation)

        if squeeze_output:
            logits = logits.squeeze(0)
            outgoing_message = outgoing_message.squeeze(0)

        if return_communication:
            return (
                logits,
                outgoing_message,
                field_ids,
                message_log_probs,
                message_entropies,
            )

        return logits

    # ======================================================
    # Communication-only forward (unchanged)
    # ======================================================

    def get_outgoing_message(self, observation: torch.Tensor):

        squeeze_output = observation.dim() == 1

        if squeeze_output:
            observation = observation.unsqueeze(0)

        local_hidden = self._get_local_hidden(observation)

        (
            field_ids,
            log_probs,
            entropies,
            communication_vector,
        ) = self.generate_communication(local_hidden)

        if squeeze_output:
            communication_vector = communication_vector.squeeze(0)
            field_ids = {k: v.squeeze(0) for k, v in field_ids.items()}
            log_probs = {k: v.squeeze(0) for k, v in log_probs.items()}
            entropies = {k: v.squeeze(0) for k, v in entropies.items()}

        return communication_vector, field_ids, log_probs, entropies

    # ======================================================
    # Local hidden (unchanged)
    # ======================================================

    def get_local_hidden(self, observation: torch.Tensor):

        squeeze_output = observation.dim() == 1

        if squeeze_output:
            observation = observation.unsqueeze(0)

        hidden = self._get_local_hidden(observation)

        if squeeze_output:
            hidden = hidden.squeeze(0)

        return hidden


# ==========================================================
# Central Critic (unchanged)
# ==========================================================

class CentralCritic(nn.Module):
    """
    Centralized critic. Unchanged -- cross-agent attention already
    models inter-agent dependency directly, which is a different
    graph (agent-level) than the local host/subnet graph above.
    """

    def __init__(self):

        super().__init__()

        self.agent_embed = nn.Linear(OBS_DIM, EMBED_DIM)

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.value_head = build_mlp(EMBED_DIM, 1)

    def forward(self, global_state: torch.Tensor):

        squeeze_output = global_state.dim() == 1

        if squeeze_output:
            global_state = global_state.unsqueeze(0)

        batch_size = global_state.shape[0]

        per_agent_obs = global_state.view(
            batch_size, NUM_AGENTS, OBS_DIM
        )

        tokens = torch.relu(self.agent_embed(per_agent_obs))

        attn_out, attention_weights = self.attention(
            tokens, tokens, tokens
        )
        self.last_attention = attention_weights.detach()

        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ffn(x))

        values = self.value_head(x).squeeze(-1)

        if squeeze_output:
            values = values.squeeze(0)

        return values


# ==========================================================
# MAPPO Model (unchanged)
# ==========================================================

class MAPPOModel(nn.Module):
    """
    Complete MAPPO model. Contains SharedActor + CentralCritic.
    """

    def __init__(
        self,
        num_targets: Optional[int] = None,
        use_host_graph: bool = True,
        subnet_edges: Optional[List[Tuple[int, int]]] = SUBNET_EDGES,
    ):

        super().__init__()

        self.actor = SharedActor(
            num_agents=NUM_AGENTS,
            communication_dim=COMMUNICATION_DIM,
            communication_latent_dim=COMMUNICATION_LATENT_DIM,
            num_targets=num_targets,
            use_host_graph=use_host_graph,
            subnet_edges=subnet_edges,
        )

        self.critic = CentralCritic()

    def act(
        self,
        observation: torch.Tensor,
        received_messages: Optional[torch.Tensor] = None,
        trust_weights: Optional[torch.Tensor] = None,
        return_communication: bool = False,
    ):

        return self.actor(
            observation,
            received_messages=received_messages,
            trust_weights=trust_weights,
            return_communication=return_communication,
        )

    def evaluate(self, global_state: torch.Tensor):

        return self.critic(global_state)
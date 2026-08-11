"""
encoder.py (v2)

Key change from the original: the old forward() consumed SOFT probability
distributions over decoder logits (softmax, not argmax) specifically to
stay differentiable end-to-end -- but that path was only ever called from
inside SharedActor.forward() where its output (outgoing_message) never
fed into the action logits. It received zero gradient regardless of being
"differentiable", so it's removed rather than kept alongside the fix, to
avoid an unused path that looks load-bearing but isn't.

encode_from_ids() is the new (and only) training-time entry point. It
takes already-sampled discrete field ids -- either freshly sampled at
rollout time, or replayed from the buffer at PPO-update time -- and embeds
them. The ids themselves are treated as constants (no gradient needed
through "which symbol was chosen": that's the decoder's job, trained via
its own log_prob). The embedding lookup and fusion MLP below ARE live,
differentiable ops, so gradient from the receiving agent's policy loss
reaches this encoder normally as long as encode_from_ids() is called
inside the same forward pass that produces that agent's action logits --
which is exactly what the SharedActor.forward() change does.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .schema import (
    ConfidenceLevel,
    EventType,
    HostStatus,
    Priority,
    TargetType,
    ThreatLevel,
)


class MessageEncoder(nn.Module):
    """Embeds a structured message (given as discrete field ids) into a fixed-size vector."""

    def __init__(
        self,
        message_dim: int = 128,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        num_targets: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.message_dim = message_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_targets = num_targets

        self.event_embedding = nn.Embedding(len(EventType), embedding_dim)
        self.target_type_embedding = nn.Embedding(len(TargetType), embedding_dim)
        self.threat_embedding = nn.Embedding(len(ThreatLevel), embedding_dim)
        self.status_embedding = nn.Embedding(len(HostStatus), embedding_dim)
        self.priority_embedding = nn.Embedding(len(Priority), embedding_dim)

        # Was a Linear(1, embedding_dim) projection of a raw sigmoid scalar.
        # confidence is now a bucketed id like everything else -- same
        # embedding-table treatment, consistent with the rest of the fields.
        self.confidence_embedding = nn.Embedding(len(ConfidenceLevel), embedding_dim)

        if num_targets is not None:
            self.target_embedding = nn.Embedding(num_targets, embedding_dim)
        else:
            self.target_embedding = None

        num_fields = 7  # event, target_type, target, threat, confidence, status, priority
        fusion_input_dim = num_fields * embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, message_dim),
        )

        self.output_norm = nn.LayerNorm(message_dim)

    def build_target_embedding(self, num_targets: int) -> None:
        if num_targets <= 0:
            raise ValueError("num_targets must be greater than zero")
        self.num_targets = num_targets
        self.target_embedding = nn.Embedding(num_targets, self.embedding_dim)

    def encode_from_ids(self, field_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Embed a message from discrete field ids of arbitrary leading shape,
        e.g. [B] for a single sender per batch item, or [B, N] for N senders
        per batch item (nn.Embedding broadcasts over any leading dims).

        Required keys: event_type, target_type, threat_level, status,
        priority, confidence. Optional: target_id (only if a target
        vocabulary has been configured).
        """
        event_vector = self.event_embedding(field_ids["event_type"])
        target_type_vector = self.target_type_embedding(field_ids["target_type"])
        threat_vector = self.threat_embedding(field_ids["threat_level"])
        status_vector = self.status_embedding(field_ids["status"])
        priority_vector = self.priority_embedding(field_ids["priority"])
        confidence_vector = self.confidence_embedding(field_ids["confidence"])

        if "target_id" in field_ids and self.target_embedding is not None:
            target_vector = self.target_embedding(field_ids["target_id"])
        else:
            target_vector = torch.zeros_like(event_vector)

        combined = torch.cat(
            [
                event_vector,
                target_type_vector,
                target_vector,
                threat_vector,
                confidence_vector,
                status_vector,
                priority_vector,
            ],
            dim=-1,
        )

        communication_vector = self.fusion(combined)
        return self.output_norm(communication_vector)
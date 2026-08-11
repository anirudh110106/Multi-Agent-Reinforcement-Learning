"""
decoder.py (v2)

Key changes from the original:

1. confidence is now a small categorical head (ConfidenceLevel, 4 buckets)
   instead of a raw sigmoid scalar. A bare continuous point-estimate has
   no valid log_prob under a policy-gradient training scheme -- everything
   transmitted needs to come from a sampleable distribution so it can be
   trained the same way as the rest of PPO.

2. Two new methods: sample_message() and evaluate_message(). These replace
   the old differentiable-argmax-free forward() as the actual training
   interface. forward() is kept because sample/evaluate call it internally,
   but it's no longer meant to be used directly by training code.

sample_message() is used at rollout time (no_grad): draws a discrete id
per field and returns the log_prob of that draw under the CURRENT policy.

evaluate_message() is used at PPO-update time: given field ids that were
sampled and stored during rollout, recomputes their log_prob under the
policy being updated. This is the exact same role Categorical(logits=...)
.log_prob(stored_action) plays for the regular environment action -- the
PPO ratio needs old vs. new log_prob of the SAME sampled value, not a new
sample.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .schema import (
    ConfidenceLevel,
    EventType,
    HostStatus,
    Priority,
    StructuredMessage,
    TargetType,
    ThreatLevel,
)

# Fields sampled the same way every time: (field_name, output_dict_key, enum_type)
_CATEGORICAL_FIELDS = (
    ("event_type", "event_logits", EventType),
    ("target_type", "target_type_logits", TargetType),
    ("threat_level", "threat_logits", ThreatLevel),
    ("status", "status_logits", HostStatus),
    ("priority", "priority_logits", Priority),
    ("confidence", "confidence_logits", ConfidenceLevel),
)


class MessageDecoder(nn.Module):
    """Decodes an agent latent representation into structured message field logits."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.event_head = nn.Linear(hidden_dim, len(EventType))
        self.target_type_head = nn.Linear(hidden_dim, len(TargetType))
        self.threat_head = nn.Linear(hidden_dim, len(ThreatLevel))
        self.status_head = nn.Linear(hidden_dim, len(HostStatus))
        self.priority_head = nn.Linear(hidden_dim, len(Priority))

        # Confidence: categorical bucket, not a sigmoid scalar. See module docstring.
        self.confidence_head = nn.Linear(hidden_dim, len(ConfidenceLevel))

        # target_id depends on the CC4 host/subnet mapping, configured later.
        self.target_head: Optional[nn.Linear] = None

    def build_target_head(self, num_targets: int) -> None:
        if num_targets <= 0:
            raise ValueError("num_targets must be greater than zero")
        self.target_head = nn.Linear(self.hidden_dim, num_targets)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Raw logits for every field. Internal use by sample_message/evaluate_message."""
        if hidden.ndim != 2:
            raise ValueError(f"Expected [batch, input_dim], got {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {hidden.shape[-1]}")

        z = self.shared(hidden)

        outputs = {
            "event_logits": self.event_head(z),
            "target_type_logits": self.target_type_head(z),
            "threat_logits": self.threat_head(z),
            "status_logits": self.status_head(z),
            "priority_logits": self.priority_head(z),
            "confidence_logits": self.confidence_head(z),
            "target_logits": self.target_head(z) if self.target_head is not None else None,
        }
        return outputs

    def sample_message(
        self, hidden: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Sample a discrete structured message from the current policy.

        Returns
        -------
        field_ids : dict[str, LongTensor[batch]]
        log_probs : dict[str, FloatTensor[batch]]
        entropies : dict[str, FloatTensor[batch]]
        """
        outputs = self.forward(hidden)

        field_ids: Dict[str, torch.Tensor] = {}
        log_probs: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, torch.Tensor] = {}

        for field, logits_key, _enum in _CATEGORICAL_FIELDS:
            dist = Categorical(logits=outputs[logits_key])
            sample = dist.sample()
            field_ids[field] = sample
            log_probs[field] = dist.log_prob(sample)
            entropies[field] = dist.entropy()

        if outputs["target_logits"] is not None:
            dist = Categorical(logits=outputs["target_logits"])
            sample = dist.sample()
            field_ids["target_id"] = sample
            log_probs["target_id"] = dist.log_prob(sample)
            entropies["target_id"] = dist.entropy()

        return field_ids, log_probs, entropies

    def evaluate_message(
        self, hidden: torch.Tensor, field_ids: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Recompute log_prob/entropy of a STORED sample under current parameters.
        Used inside the PPO update -- same role as
        Categorical(logits=...).log_prob(stored_action) for the env action.
        """
        outputs = self.forward(hidden)

        log_probs: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, torch.Tensor] = {}

        for field, logits_key, _enum in _CATEGORICAL_FIELDS:
            dist = Categorical(logits=outputs[logits_key])
            log_probs[field] = dist.log_prob(field_ids[field])
            entropies[field] = dist.entropy()

        if outputs["target_logits"] is not None and "target_id" in field_ids:
            dist = Categorical(logits=outputs["target_logits"])
            log_probs["target_id"] = dist.log_prob(field_ids["target_id"])
            entropies["target_id"] = dist.entropy()

        return log_probs, entropies

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor) -> StructuredMessage:
        """Hard argmax decode for inference/debugging/logging only. Not used in training."""
        if hidden.ndim != 1:
            raise ValueError("decode() expects a single vector with shape [input_dim]")

        outputs = self.forward(hidden.unsqueeze(0))

        event_type = EventType(torch.argmax(outputs["event_logits"][0]).item())
        target_type = TargetType(torch.argmax(outputs["target_type_logits"][0]).item())
        threat_level = ThreatLevel(torch.argmax(outputs["threat_logits"][0]).item())
        status = HostStatus(torch.argmax(outputs["status_logits"][0]).item())
        priority = Priority(torch.argmax(outputs["priority_logits"][0]).item())

        confidence_level = ConfidenceLevel(torch.argmax(outputs["confidence_logits"][0]).item())
        confidence = float(confidence_level) / (len(ConfidenceLevel) - 1)  # bucket -> [0,1] for evaluator.py

        if outputs["target_logits"] is None:
            target_id = 0
        else:
            target_id = torch.argmax(outputs["target_logits"][0]).item()

        return StructuredMessage(
            event_type=event_type,
            target_type=target_type,
            target_id=target_id,
            threat_level=threat_level,
            confidence=confidence,
            status=status,
            priority=priority,
        )
"""
schema.py

Structured communication schema for CC4 MARL.

This file defines the semantic message exchanged between Blue agents.
It does NOT perform neural-network encoding or decoding.

Communication flow:

    Agent hidden state
          |
          v
    Message Decoder
          |
          v
    StructuredMessage
          |
          v
    Message Encoder
          |
          v
    Communication Vector
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Message field definitions
# ---------------------------------------------------------------------------

class EventType(IntEnum):
    """
    Type of cyber-security event being communicated.

    Values are intentionally integer-based because these fields will
    eventually be represented by categorical neural-network outputs.
    """

    NONE = 0
    DISCOVERY = 1
    SCAN = 2
    SUSPICIOUS_ACTIVITY = 3
    COMPROMISE = 4
    LATERAL_MOVEMENT = 5
    PRIVILEGE_ESCALATION = 6
    RECOVERY = 7


class ThreatLevel(IntEnum):
    """
    Severity of the reported situation.

    Ordered from least severe to most severe.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class ConfidenceLevel(IntEnum):
    """
    Discrete confidence levels used by the message policy.

    The decoder samples one of these categories during
    communication-policy execution.
    """

    VERY_LOW = 0
    LOW = 1
    HIGH = 2
    VERY_HIGH = 3


class HostStatus(IntEnum):
    """
    Reported status of the target.

    """

    UNKNOWN = 0
    NORMAL = 1
    SUSPICIOUS = 2
    COMPROMISED = 3
    CONTAINED = 4


class Priority(IntEnum):
    """
    Urgency of the communicated information.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    URGENT = 3


class TargetType(IntEnum):
    """
    Type of entity being referenced by the message.
    """

    NONE = 0
    HOST = 1
    SUBNET = 2


# ---------------------------------------------------------------------------
# Structured message
# ---------------------------------------------------------------------------

@dataclass
class StructuredMessage:
    """
    Semantic message exchanged between Blue agents.

    This is the intermediate representation between the neural-network
    latent vector and the communication vector.

    Example
    -------
    StructuredMessage(
        event_type=EventType.COMPROMISE,
        target_type=TargetType.HOST,
        target_id=12,
        threat_level=ThreatLevel.HIGH,
        confidence=0.91,
        status=HostStatus.COMPROMISED,
        priority=Priority.URGENT,
    )
    """

    # What happened?
    event_type: EventType = EventType.NONE

    # What is affected?
    target_type: TargetType = TargetType.NONE

    # Numerical identifier for the target.
    #
    # The interpretation depends on target_type:
    #   HOST   -> host index
    #   SUBNET -> subnet index
    #   NONE   -> ignored
    target_id: int = 0

    # How serious is the event?
    threat_level: ThreatLevel = ThreatLevel.LOW

    # How confident is the sender?
    #
    # Continuous value in [0, 1].
    confidence: float = 0.0

    # Current state of the target.
    status: HostStatus = HostStatus.UNKNOWN

    # How urgently should another agent react?
    priority: Priority = Priority.LOW

    def __post_init__(self) -> None:
        """Validate and normalize the structured message."""

        self.event_type = EventType(self.event_type)
        self.target_type = TargetType(self.target_type)
        self.threat_level = ThreatLevel(self.threat_level)
        self.status = HostStatus(self.status)
        self.priority = Priority(self.priority)

        self.target_id = int(self.target_id)

        # Confidence must always be a valid probability.
        self.confidence = float(self.confidence)
        self.confidence = max(0.0, min(1.0, self.confidence))

        if self.target_id < 0:
            raise ValueError("target_id must be >= 0")

    @classmethod
    def empty(cls) -> "StructuredMessage":
        """
        Return an empty/no-information message.

        Used when an agent has nothing useful to communicate.
        """

        return cls()

    def is_empty(self) -> bool:
        """Return True when the message contains no meaningful event."""

        return self.event_type == EventType.NONE

    def as_dict(self) -> dict:
        """
        Convert the message to a human-readable dictionary.

        Useful for logging, debugging and evaluation.
        """

        return {
            "event_type": self.event_type.name,
            "target_type": self.target_type.name,
            "target_id": self.target_id,
            "threat_level": self.threat_level.name,
            "confidence": self.confidence,
            "status": self.status.name,
            "priority": self.priority.name,
        }

    def __repr__(self) -> str:
        fields = self.as_dict()

        return (
            "StructuredMessage("
            f"event_type={fields['event_type']}, "
            f"target_type={fields['target_type']}, "
            f"target_id={fields['target_id']}, "
            f"threat_level={fields['threat_level']}, "
            f"confidence={fields['confidence']:.3f}, "
            f"status={fields['status']}, "
            f"priority={fields['priority']}"
            ")"
        )


# ---------------------------------------------------------------------------
# Schema metadata
# ---------------------------------------------------------------------------

MESSAGE_FIELDS = (
    "event_type",
    "target_type",
    "target_id",
    "threat_level",
    "confidence",
    "status",
    "priority",
)


MESSAGE_SCHEMA = {
    "event_type": {
        "type": "categorical",
        "size": len(EventType),
    },
    "target_type": {
        "type": "categorical",
        "size": len(TargetType),
    },
    "target_id": {
        "type": "categorical",
        "size": None,  # Determined from CC4 host/subnet mapping.
    },
    "threat_level": {
        "type": "ordinal",
        "size": len(ThreatLevel),
    },
    # "confidence": {
    #     "type": "continuous",
    #     "range": (0.0, 1.0),
    # },

    "confidence": {
        "type": "categorical",
        "size": len(ConfidenceLevel),
    },
    "status": {
        "type": "categorical",
        "size": len(HostStatus),
    },
    "priority": {
        "type": "ordinal",
        "size": len(Priority),
    },
}


def empty_message() -> StructuredMessage:
    """Convenience function for creating an empty message."""

    return StructuredMessage.empty()
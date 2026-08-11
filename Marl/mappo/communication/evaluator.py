"""
evaluator.py

Message quality evaluator for structured communication in CC4 MARL.

The evaluator determines how correct/useful a message was after the
environment produces the next state.

Flow:

    Sender
       |
       v
    StructuredMessage
       |
       v
    Receiver
       |
       v
    Environment step
       |
       v
    Ground-truth state
       |
       v
    MessageEvaluator
       |
       v
    message_quality [0, 1]
       |
       v
    DynamicTrust.update()

IMPORTANT
---------
The ground-truth state used here is training/evaluation-side information.
It must NOT be provided to the Blue agents as part of their observation.

This module does NOT:
    - modify trust
    - encode messages
    - decode messages
    - interact directly with CybORG
    - perform MAPPO updates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .schema import (
    EventType,
    HostStatus,
    Priority,
    StructuredMessage,
    TargetType,
    ThreatLevel,
)


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class MessageEvaluation:
    """
    Result of evaluating one structured message.

    All scores are in [0, 1].

    Attributes
    ----------
    overall_score:
        Final message quality sent to the trust mechanism.

    event_score:
        Correctness of the reported event.

    target_score:
        Correctness of the reported target.

    threat_score:
        Correctness of the reported threat level.

    status_score:
        Correctness of the reported status.

    usefulness_score:
        Whether the information was operationally useful.

    confidence:
        Confidence declared by the sender.

    details:
        Human-readable debugging information.
    """

    overall_score: float

    event_score: float
    target_score: float
    threat_score: float
    status_score: float
    usefulness_score: float

    confidence: float

    details: Optional[Dict[str, Any]] = None

    def as_dict(self) -> dict:
        """Return the evaluation as a dictionary."""

        return {
            "overall_score": self.overall_score,
            "event_score": self.event_score,
            "target_score": self.target_score,
            "threat_score": self.threat_score,
            "status_score": self.status_score,
            "usefulness_score": self.usefulness_score,
            "confidence": self.confidence,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Message evaluator
# ---------------------------------------------------------------------------

class MessageEvaluator:
    """
    Evaluates structured cyber-security messages.

    The evaluator compares the sender's message against ground-truth
    information available to the training/evaluation process.

    Parameters
    ----------
    event_weight:
        Weight assigned to event correctness.

    target_weight:
        Weight assigned to target correctness.

    threat_weight:
        Weight assigned to threat-level correctness.

    status_weight:
        Weight assigned to status correctness.

    usefulness_weight:
        Weight assigned to operational usefulness.

    Notes
    -----
    The default weights sum to 1.0.

    The target receives a relatively high weight because a message
    identifying the wrong host/subnet is significantly less useful
    in a cyber-defense setting.
    """

    def __init__(
        self,
        event_weight: float = 0.25,
        target_weight: float = 0.30,
        threat_weight: float = 0.15,
        status_weight: float = 0.15,
        usefulness_weight: float = 0.15,
    ) -> None:

        weights = [
            event_weight,
            target_weight,
            threat_weight,
            status_weight,
            usefulness_weight,
        ]

        if any(weight < 0.0 for weight in weights):
            raise ValueError(
                "Evaluation weights must be non-negative."
            )

        total = sum(weights)

        if total <= 0.0:
            raise ValueError(
                "At least one evaluation weight must be positive."
            )

        # Normalize automatically.
        self.event_weight = event_weight / total
        self.target_weight = target_weight / total
        self.threat_weight = threat_weight / total
        self.status_weight = status_weight / total
        self.usefulness_weight = usefulness_weight / total

    # ------------------------------------------------------------------
    # Public evaluation interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> MessageEvaluation:
        """
        Evaluate a structured message against ground truth.

        Parameters
        ----------
        message:
            Structured message sent by another Blue agent.

        ground_truth:
            Ground-truth information relevant to the message.

            Expected keys can include:

                event_type
                target_type
                target_id
                threat_level
                status
                compromised
                suspicious
                useful

            The exact contents can be adapted to the CC4 state
            representation.

        previous_state:
            Optional state before the environment step.

        current_state:
            Optional state after the environment step.

        Returns
        -------
        MessageEvaluation
            Evaluation result containing a quality score in [0, 1].
        """

        if not isinstance(message, StructuredMessage):
            raise TypeError(
                "message must be a StructuredMessage."
            )

        if not isinstance(ground_truth, dict):
            raise TypeError(
                "ground_truth must be a dictionary."
            )

        event_score = self._evaluate_event(
            message,
            ground_truth,
        )

        target_score = self._evaluate_target(
            message,
            ground_truth,
        )

        threat_score = self._evaluate_threat(
            message,
            ground_truth,
        )

        status_score = self._evaluate_status(
            message,
            ground_truth,
        )

        usefulness_score = self._evaluate_usefulness(
            message,
            ground_truth,
            previous_state,
            current_state,
        )

        overall_score = (
            self.event_weight * event_score
            + self.target_weight * target_score
            + self.threat_weight * threat_score
            + self.status_weight * status_score
            + self.usefulness_weight * usefulness_score
        )

        overall_score = self._clamp(
            overall_score
        )

        return MessageEvaluation(
            overall_score=overall_score,
            event_score=event_score,
            target_score=target_score,
            threat_score=threat_score,
            status_score=status_score,
            usefulness_score=usefulness_score,
            confidence=message.confidence,
            details={
                "message": message.as_dict(),
                "ground_truth": ground_truth,
            },
        )

    # ------------------------------------------------------------------
    # Event evaluation
    # ------------------------------------------------------------------

    def _evaluate_event(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
    ) -> float:
        """
        Evaluate event-type correctness.

        Exact match gives 1.0.

        If the ground truth does not contain event information,
        the evaluator returns 0.5 rather than falsely declaring
        the message incorrect.
        """

        actual = ground_truth.get("event_type")

        if actual is None:
            return 0.5

        actual = self._normalize_enum(
            actual,
            EventType,
        )

        if actual is None:
            return 0.5

        if message.event_type == actual:
            return 1.0

        # NONE is particularly bad when an actual event exists.
        if message.event_type == EventType.NONE:
            return 0.0

        # Some cyber events are semantically related.
        related_events = {
            EventType.DISCOVERY: {
                EventType.SCAN,
                EventType.DISCOVERY,
            },
            EventType.SCAN: {
                EventType.SCAN,
                EventType.DISCOVERY,
            },
            EventType.SUSPICIOUS_ACTIVITY: {
                EventType.SUSPICIOUS_ACTIVITY,
                EventType.DISCOVERY,
                EventType.SCAN,
            },
            EventType.COMPROMISE: {
                EventType.COMPROMISE,
                EventType.PRIVILEGE_ESCALATION,
            },
            EventType.PRIVILEGE_ESCALATION: {
                EventType.PRIVILEGE_ESCALATION,
                EventType.COMPROMISE,
            },
            EventType.LATERAL_MOVEMENT: {
                EventType.LATERAL_MOVEMENT,
            },
            EventType.RECOVERY: {
                EventType.RECOVERY,
            },
        }

        if actual in related_events.get(
            message.event_type,
            set(),
        ):
            return 0.5

        return 0.0

    # ------------------------------------------------------------------
    # Target evaluation
    # ------------------------------------------------------------------

    def _evaluate_target(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
    ) -> float:
        """
        Evaluate target correctness.

        Target is treated as highly important because a correct
        threat report about the wrong host is not operationally
        equivalent to a correct target identification.
        """

        actual_type = ground_truth.get(
            "target_type"
        )

        actual_id = ground_truth.get(
            "target_id"
        )

        if actual_type is None and actual_id is None:
            return 0.5

        score = 0.0

        if actual_type is not None:

            actual_type = self._normalize_enum(
                actual_type,
                TargetType,
            )

            if actual_type is not None:
                if message.target_type == actual_type:
                    score += 0.5

        if actual_id is not None:

            try:
                actual_id = int(actual_id)
            except (TypeError, ValueError):
                actual_id = None

            if actual_id is not None:
                if message.target_id == actual_id:
                    score += 0.5

        return score

    # ------------------------------------------------------------------
    # Threat evaluation
    # ------------------------------------------------------------------

    def _evaluate_threat(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
    ) -> float:
        """
        Evaluate threat-level correctness.

        Exact level:
            1.0

        One level away:
            0.5

        Two or more levels away:
            0.0

        This is better than strict binary accuracy because predicting
        HIGH instead of CRITICAL is not equivalent to predicting LOW.
        """

        actual = ground_truth.get(
            "threat_level"
        )

        if actual is None:
            return 0.5

        actual = self._normalize_enum(
            actual,
            ThreatLevel,
        )

        if actual is None:
            return 0.5

        predicted = int(
            message.threat_level
        )

        actual = int(actual)

        difference = abs(
            predicted - actual
        )

        if difference == 0:
            return 1.0

        if difference == 1:
            return 0.5

        return 0.0

    # ------------------------------------------------------------------
    # Status evaluation
    # ------------------------------------------------------------------

    def _evaluate_status(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
    ) -> float:
        """Evaluate host/status correctness."""

        actual = ground_truth.get(
            "status"
        )

        # Support a simpler ground-truth representation.
        if actual is None:

            if ground_truth.get(
                "compromised"
            ) is True:
                actual = HostStatus.COMPROMISED

            elif ground_truth.get(
                "suspicious"
            ) is True:
                actual = HostStatus.SUSPICIOUS

            elif (
                "compromised" in ground_truth
                or "suspicious" in ground_truth
            ):
                actual = HostStatus.NORMAL

        if actual is None:
            return 0.5

        actual = self._normalize_enum(
            actual,
            HostStatus,
        )

        if actual is None:
            return 0.5

        if message.status == actual:
            return 1.0

        # Unknown is less informative than an actual state.
        if message.status == HostStatus.UNKNOWN:
            return 0.0

        # NORMAL vs SUSPICIOUS is partially related.
        if {
            message.status,
            actual,
        } == {
            HostStatus.NORMAL,
            HostStatus.SUSPICIOUS,
        }:
            return 0.5

        return 0.0

    # ------------------------------------------------------------------
    # Usefulness evaluation
    # ------------------------------------------------------------------

    def _evaluate_usefulness(
        self,
        message: StructuredMessage,
        ground_truth: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]],
        current_state: Optional[Dict[str, Any]],
    ) -> float:
        """
        Evaluate operational usefulness.

        Priority order:

        1. Explicit evaluator signal, if supplied.
        2. Whether the message caused a useful state change.
        3. Whether the message contains meaningful information.
        """

        # --------------------------------------------------------------
        # Explicit usefulness signal
        # --------------------------------------------------------------

        explicit_usefulness = ground_truth.get(
            "useful"
        )

        if explicit_usefulness is not None:

            try:
                return self._clamp(
                    float(explicit_usefulness)
                )
            except (TypeError, ValueError):
                pass

        # --------------------------------------------------------------
        # State-transition based usefulness
        # --------------------------------------------------------------

        if (
            previous_state is not None
            and current_state is not None
        ):

            state_change = self._measure_state_change(
                previous_state,
                current_state,
            )

            if state_change > 0.0:

                # A message associated with a meaningful environment
                # change receives a positive usefulness score.
                return self._clamp(
                    state_change
                )

        # --------------------------------------------------------------
        # Fallback
        # --------------------------------------------------------------

        # A message containing no event is not useful.
        if message.event_type == EventType.NONE:
            return 0.0

        # A meaningful structured message gets a neutral-positive
        # baseline until explicit operational feedback is available.
        return 0.5

    # ------------------------------------------------------------------
    # State change
    # ------------------------------------------------------------------

    def _measure_state_change(
        self,
        previous_state: Dict[str, Any],
        current_state: Dict[str, Any],
    ) -> float:
        """
        Estimate whether the environment changed in a meaningful way.

        This is intentionally conservative.

        The current implementation looks for common CC4-style
        indicators such as:

            compromised
            suspicious
            active_sessions
            malicious_processes
            network_connections

        The function can later be replaced by a CC4-specific evaluator
        once we define exactly which state transitions correspond to
        useful defensive outcomes.
        """

        indicators = [
            "compromised",
            "suspicious",
            "active_sessions",
            "malicious_processes",
            "network_connections",
        ]

        changed = 0
        available = 0

        for key in indicators:

            if (
                key not in previous_state
                or key not in current_state
            ):
                continue

            available += 1

            if (
                previous_state[key]
                != current_state[key]
            ):
                changed += 1

        if available == 0:
            return 0.0

        return changed / available

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def confidence_adjusted_score(
        self,
        evaluation: MessageEvaluation,
    ) -> float:
        """
        Produce a confidence-aware message quality score.

        The confidence is NOT used as proof of correctness.

        Instead, highly confident incorrect messages are penalized
        more strongly than uncertain incorrect messages.

        Correctness remains the dominant factor.
        """

        quality = evaluation.overall_score
        confidence = evaluation.confidence

        # Distance from a neutral confidence level.
        confidence_strength = abs(
            confidence - 0.5
        ) * 2.0

        if quality >= 0.5:
            # Correct information benefits slightly from confidence.
            adjusted = (
                quality
                + 0.10
                * confidence_strength
                * quality
            )
        else:
            # Incorrect high-confidence information is more damaging.
            penalty = (
                0.10
                * confidence_strength
                * (1.0 - quality)
            )

            adjusted = quality - penalty

        return self._clamp(
            adjusted
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """Clamp a scalar to [minimum, maximum]."""

        return max(
            minimum,
            min(maximum, float(value)),
        )

    @staticmethod
    def _normalize_enum(
        value: Any,
        enum_type,
    ):
        """
        Convert an arbitrary enum representation into the
        corresponding IntEnum.

        Supports:

            IntEnum
            integer
            enum name string
        """

        if isinstance(
            value,
            enum_type,
        ):
            return value

        if isinstance(
            value,
            str,
        ):

            try:
                return enum_type[
                    value.upper()
                ]
            except KeyError:
                return None

        try:
            return enum_type(
                int(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            return None
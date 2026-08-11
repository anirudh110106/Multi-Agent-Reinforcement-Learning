"""
trust.py

Dynamic trust mechanism for structured communication between
the five Blue agents in CC4.

Trust is directional:

    trust[sender][receiver]

means:

    "How much does receiver trust sender?"

Trust is updated from the quality of information communicated
by the sender.

The mechanism uses a Beta-Bernoulli style reputation model with
exponential forgetting.

Conceptually:

    correct/useful message -> increase trust
    incorrect/useless message -> decrease trust

Recent evidence has more influence than old evidence.

This module does NOT:
    - decode messages
    - encode messages
    - generate messages
    - interact with CybORG
    - perform MAPPO updates
"""

from __future__ import annotations

from typing import Optional

import torch


class DynamicTrust:
    """
    Maintains a directed dynamic trust matrix between agents.

    Parameters
    ----------
    num_agents : int
        Number of communicating agents.

    prior_alpha : float
        Initial positive/correct evidence.

    prior_beta : float
        Initial negative/incorrect evidence.

    decay : float
        Exponential forgetting factor.

        1.0:
            no forgetting.

        Values below 1.0:
            older evidence gradually loses influence.

    min_trust : float
        Lower bound for trust.

    max_trust : float
        Upper bound for trust.

    Notes
    -----
    For sender i and receiver j:

        trust[i][j] =
            alpha[i][j] /
            (alpha[i][j] + beta[i][j])

    where:

        alpha = evidence for useful/correct information
        beta  = evidence for incorrect/useless information

    The matrix is directional.

        trust[A][B] != trust[B][A]

    is completely valid.
    """

    def __init__(
        self,
        num_agents: int = 5,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay: float = 0.995,
        min_trust: float = 0.05,
        max_trust: float = 0.95,
    ) -> None:

        if num_agents <= 0:
            raise ValueError(
                "num_agents must be greater than zero."
            )

        if prior_alpha <= 0:
            raise ValueError(
                "prior_alpha must be greater than zero."
            )

        if prior_beta <= 0:
            raise ValueError(
                "prior_beta must be greater than zero."
            )

        if not 0.0 < decay <= 1.0:
            raise ValueError(
                "decay must be in the range (0, 1]."
            )

        if not 0.0 <= min_trust < max_trust <= 1.0:
            raise ValueError(
                "Trust bounds must satisfy "
                "0 <= min_trust < max_trust <= 1."
            )

        self.num_agents = num_agents

        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

        self.decay = float(decay)

        self.min_trust = float(min_trust)
        self.max_trust = float(max_trust)

        # ---------------------------------------------------------------
        # Evidence matrices
        # ---------------------------------------------------------------

        # alpha[i][j]:
        # positive/useful evidence for sender i as judged by receiver j.
        self.alpha = torch.full(
            (num_agents, num_agents),
            self.prior_alpha,
            dtype=torch.float32,
        )

        # beta[i][j]:
        # negative/useless evidence for sender i as judged by receiver j.
        self.beta = torch.full(
            (num_agents, num_agents),
            self.prior_beta,
            dtype=torch.float32,
        )

        # An agent does not need to trust itself.
        #
        # We keep the diagonal at zero and exclude it from normal
        # communication/trust calculations.
        self._zero_diagonal()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_diagonal(self) -> None:
        """Remove self-trust entries from the evidence matrices."""

        indices = torch.arange(self.num_agents)

        self.alpha[indices, indices] = 0.0
        self.beta[indices, indices] = 0.0

    def _validate_agent_pair(
        self,
        sender: int,
        receiver: int,
    ) -> None:
        """Validate sender and receiver IDs."""

        if not 0 <= sender < self.num_agents:
            raise IndexError(
                f"sender must be in "
                f"[0, {self.num_agents - 1}], got {sender}"
            )

        if not 0 <= receiver < self.num_agents:
            raise IndexError(
                f"receiver must be in "
                f"[0, {self.num_agents - 1}], got {receiver}"
            )

        if sender == receiver:
            raise ValueError(
                "sender and receiver must be different."
            )

    # ------------------------------------------------------------------
    # Trust calculation
    # ------------------------------------------------------------------

    def get_trust(
        self,
        sender: int,
        receiver: int,
    ) -> float:
        """
        Return the current trust score.

        Parameters
        ----------
        sender : int
            Agent that sent the message.

        receiver : int
            Agent evaluating the message.

        Returns
        -------
        float
            Trust score in [min_trust, max_trust].
        """

        self._validate_agent_pair(
            sender,
            receiver,
        )

        alpha = self.alpha[sender, receiver]
        beta = self.beta[sender, receiver]

        raw_trust = alpha / (alpha + beta)

        return float(
            torch.clamp(
                raw_trust,
                self.min_trust,
                self.max_trust,
            )
        )

    def get_trust_matrix(self) -> torch.Tensor:
        """
        Return the complete directed trust matrix.

        Matrix interpretation:

            matrix[sender][receiver]

        Example:

            matrix[0][2]

        means:

            Agent 2's trust in Agent 0.
        """

        denominator = self.alpha + self.beta

        trust = torch.zeros_like(
            denominator
        )

        valid = denominator > 0

        trust[valid] = (
            self.alpha[valid]
            / denominator[valid]
        )

        trust = torch.clamp(
            trust,
            self.min_trust,
            self.max_trust,
        )

        # No self-trust.
        indices = torch.arange(self.num_agents)
        trust[indices, indices] = 0.0

        return trust

    # ------------------------------------------------------------------
    # Updating trust
    # ------------------------------------------------------------------

    def update(
        self,
        sender: int,
        receiver: int,
        correctness: float,
        weight: float = 1.0,
    ) -> float:
        """
        Update trust based on the quality of a received message.

        Parameters
        ----------
        sender : int
            Agent that sent the message.

        receiver : int
            Agent evaluating the message.

        correctness : float
            Quality of the information.

            1.0 -> completely correct/useful
            0.0 -> completely incorrect/useless

            Values between 0 and 1 are allowed.

        weight : float
            Importance of this particular message.

            Higher weight means the message has more influence
            on the trust update.

        Returns
        -------
        float
            Updated trust score.

        Examples
        --------
        Correct message:

            update(0, 1, correctness=1.0)

        Incorrect message:

            update(0, 1, correctness=0.0)

        Partially useful message:

            update(0, 1, correctness=0.6)
        """

        self._validate_agent_pair(
            sender,
            receiver,
        )

        correctness = float(correctness)
        weight = float(weight)

        if not 0.0 <= correctness <= 1.0:
            raise ValueError(
                "correctness must be in the range [0, 1]."
            )

        if weight <= 0.0:
            raise ValueError(
                "weight must be greater than zero."
            )

        # ---------------------------------------------------------------
        # Forget old evidence
        # ---------------------------------------------------------------

        self.alpha[sender, receiver] *= self.decay
        self.beta[sender, receiver] *= self.decay

        # ---------------------------------------------------------------
        # Add new evidence
        # ---------------------------------------------------------------

        # A correctness of 1.0 contributes entirely to alpha.
        #
        # A correctness of 0.0 contributes entirely to beta.
        #
        # Intermediate correctness distributes evidence between them.
        self.alpha[sender, receiver] += (
            weight * correctness
        )

        self.beta[sender, receiver] += (
            weight * (1.0 - correctness)
        )

        return self.get_trust(
            sender,
            receiver,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def record_correct(
        self,
        sender: int,
        receiver: int,
        weight: float = 1.0,
    ) -> float:
        """
        Record a completely correct/useful message.
        """

        return self.update(
            sender=sender,
            receiver=receiver,
            correctness=1.0,
            weight=weight,
        )

    def record_incorrect(
        self,
        sender: int,
        receiver: int,
        weight: float = 1.0,
    ) -> float:
        """
        Record a completely incorrect/useless message.
        """

        return self.update(
            sender=sender,
            receiver=receiver,
            correctness=0.0,
            weight=weight,
        )

    def record_partial(
        self,
        sender: int,
        receiver: int,
        correctness: float,
        weight: float = 1.0,
    ) -> float:
        """
        Record a partially useful message.
        """

        return self.update(
            sender=sender,
            receiver=receiver,
            correctness=correctness,
            weight=weight,
        )

    # ------------------------------------------------------------------
    # Message weighting
    # ------------------------------------------------------------------

    def weight_message(
        self,
        sender: int,
        receiver: int,
        message_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the receiver's trust in the sender to a message vector.

        Parameters
        ----------
        sender : int
            Sending agent.

        receiver : int
            Receiving agent.

        message_vector : torch.Tensor
            Encoded communication vector.

        Returns
        -------
        torch.Tensor
            Trust-weighted communication vector.

        Formula:

            weighted_message =
                trust(sender -> receiver)
                * message_vector
        """

        trust = self.get_trust(
            sender,
            receiver,
        )

        return message_vector * trust

    def get_trust_weights(
        self,
        receiver: int,
    ) -> torch.Tensor:
        """
        Return all sender trust scores for one receiver.

        Parameters
        ----------
        receiver : int
            Agent receiving the messages.

        Returns
        -------
        torch.Tensor
            Shape [num_agents].

            weights[sender] =
                trust(sender -> receiver)
        """

        if not 0 <= receiver < self.num_agents:
            raise IndexError(
                f"receiver must be in "
                f"[0, {self.num_agents - 1}]"
            )

        matrix = self.get_trust_matrix()

        return matrix[:, receiver]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all trust relationships to the initial prior.
        """

        self.alpha.fill_(
            self.prior_alpha
        )

        self.beta.fill_(
            self.prior_beta
        )

        self._zero_diagonal()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """
        Return trust state for checkpointing.
        """

        return {
            "alpha": self.alpha.clone(),
            "beta": self.beta.clone(),
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        """
        Restore trust state from a checkpoint.
        """

        if "alpha" not in state:
            raise KeyError(
                "Trust state is missing 'alpha'."
            )

        if "beta" not in state:
            raise KeyError(
                "Trust state is missing 'beta'."
            )

        if state["alpha"].shape != (
            self.num_agents,
            self.num_agents,
        ):
            raise ValueError(
                "Invalid alpha matrix shape."
            )

        if state["beta"].shape != (
            self.num_agents,
            self.num_agents,
        ):
            raise ValueError(
                "Invalid beta matrix shape."
            )

        self.alpha = state["alpha"].clone()
        self.beta = state["beta"].clone()

        self._zero_diagonal()






# help gng

# StructuredMessage
#        │
#        ▼
# MessageEvaluator
#        │
#        ├── factual correctness
#        ├── target correctness
#        ├── threat/status correctness
#        └── operational usefulness
#        │
#        ▼
# message_quality ∈ [0,1]
#        │
#        ▼
# DynamicTrust.update()
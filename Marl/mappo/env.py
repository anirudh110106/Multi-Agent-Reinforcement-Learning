"""
env.py

Environment interface for MAPPO.

This file is the ONLY place that interacts with CybORG.
The rest of the project only imports CC4Env.

Ground truth
------------

get_ground_truth() is a heuristic, not a perfect oracle: for a given
sender agent, it checks whether any red-agent session exists on a host
within that agent's assigned subnets, per the CURRENT true state. This
is enough to give DynamicTrust a real (non-null) correctness signal --
"did this agent report a compromise when one actually existed in its
zone" -- but it does not yet distinguish SCAN vs COMPROMISE vs
LATERAL_MOVEMENT, or identify a specific target host. Refine using
state.hosts[h].events (old_process_creation / process_creation,
old_network_connections / network_connections -- see
BlueFlatWrapper._get_procesess / _get_connections for the exact
pattern) if you need finer-grained event_type ground truth later.
"""

from CybORG import CybORG
from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE

from .config import EPISODE_LENGTH
# from .communication.schema import EventType, HostStatus, ThreatLevel
from .communication.schema import (
    EventType,
    HostStatus,
    ThreatLevel,
    TargetType,
)


class CC4Env:

    def get_num_targets(self):
            """
            Return the number of possible communication targets.

            Target IDs use the deterministic mapping:

                sorted(state.hosts.keys())

            Therefore the target vocabulary size is the number
            of hosts currently known to CybORG.
            """

            state = self.cyborg.environment_controller.state

            hostnames = sorted(state.hosts.keys())

            return len(hostnames)

    def __init__(self, red_agent_class=FiniteStateRedAgent):

        scenario = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=red_agent_class,
            steps=EPISODE_LENGTH,
        )

        cyborg = CybORG(scenario_generator=scenario)

        # Kept so get_ground_truth() (and anything else that needs true
        # state) can reach environment_controller.state directly, the
        # same way TrueStateTableWrapper does. Previously this was a
        # local variable in __init__ and was lost after construction.
        self.cyborg = cyborg

        # Official CC4 MARL wrapper
        self.env = EnterpriseMAE(cyborg)

        self.agent_names = list(self.env.agents)

    ############################################################

    def reset(self, seed=None):

        observations, info = self.env.reset(seed=seed)

        return observations, info

    ############################################################

    def step(self, actions, messages=None):

        if messages is None:
            messages = {}

        observations, rewards, terminated, truncated, info = \
            self.env.step(actions, messages)

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    @property
    def agents(self):

        return self.env.agents

    @property
    def possible_agents(self):

        return self.env.possible_agents

    def observation_space(self, agent):

        return self.env.observation_space(agent)

    def action_space(self, agent):

        return self.env.action_space(agent)

    def sample_actions(self):
        """
        Random action for every blue agent.
        Useful for testing.
        """

        actions = {}

        for agent in self.agents:
            actions[agent] = self.action_space(agent).sample()

        return actions

    def get_observation_dims(self):
        dims = {}
        for agent in self.agents:
            dims[agent] = self.observation_space(agent).shape[0]

        return dims

    ############################################################

    def get_action_dims(self):

        dims = {}

        for agent in self.agents:
            dims[agent] = self.action_space(agent).n

        return dims

    ############################################################
    # Ground truth for MessageEvaluator / DynamicTrust
    ############################################################

    def get_ground_truth(
            self,
            sender_id,
            receiver_id,
            message,
            previous_info,
            current_info,
        ):
            """
            Training-side ground truth for one sender's structured message.

            The ground truth is derived from the CURRENT true CybORG state.

            For the sender's assigned subnets, this function identifies
            compromised hosts containing an active Red session.

            Returns information compatible with MessageEvaluator:

                event_type
                target_type
                target_id
                threat_level
                status
                compromised
                suspicious

            target_id uses the deterministic sorted host list:

                sorted(state.hosts.keys())

            This MUST remain consistent with the target-ID mapping used by
            the communication encoder/decoder.
            """

            # ------------------------------------------------------------
            # Resolve sender
            # ------------------------------------------------------------

            agent_names = sorted(self.possible_agents)

            if not (0 <= sender_id < len(agent_names)):
                return None

            sender_name = agent_names[sender_id]

            # ------------------------------------------------------------
            # Access true CybORG state
            # ------------------------------------------------------------

            state = self.cyborg.environment_controller.state

            agent_meta = state.scenario.agents.get(
                sender_name
            )

            if agent_meta is None:
                return None

            # ------------------------------------------------------------
            # Sender's assigned subnets
            # ------------------------------------------------------------

            sender_subnets = {
                str(subnet).lower()
                for subnet in agent_meta.allowed_subnets
            }

            # ------------------------------------------------------------
            # Deterministic host -> target_id mapping
            #
            # IMPORTANT:
            # This mapping must match the mapping used by the
            # communication module.
            # ------------------------------------------------------------

            hostnames = sorted(
                state.hosts.keys()
            )

            host_to_id = {
                hostname: index
                for index, hostname in enumerate(hostnames)
            }

            # ------------------------------------------------------------
            # Find compromised hosts in sender's zone
            # ------------------------------------------------------------

            compromised_hosts = []

            for hostname, host in state.hosts.items():

                subnet = state.hostname_subnet_map.get(
                    hostname
                )

                if subnet is None:
                    continue

                if str(subnet).lower() not in sender_subnets:
                    continue

                sessions = getattr(
                    host,
                    "sessions",
                    {},
                )

                red_session_found = False

                for owner, session_list in sessions.items():

                    if (
                        "red" in str(owner).lower()
                        and session_list
                    ):
                        red_session_found = True
                        break

                if red_session_found:
                    compromised_hosts.append(
                        hostname
                    )

            # ------------------------------------------------------------
            # No compromise in sender's zone
            # ------------------------------------------------------------

            if not compromised_hosts:

                return {
                    "event_type": EventType.NONE,

                    "target_type": TargetType.NONE,

                    "target_id": 0,

                    "threat_level": ThreatLevel.LOW,

                    "status": HostStatus.NORMAL,

                    "compromised": False,

                    "suspicious": False,

                    "useful": False,
                }

            # ------------------------------------------------------------
            # Compromise exists
            #
            # For now we use the first deterministic compromised host.
            #
            # Later we can upgrade this to evaluate multiple targets.
            # ------------------------------------------------------------

            compromised_hosts.sort()

            target_hostname = compromised_hosts[0]

            target_id = host_to_id[
                target_hostname
            ]

            return {
                "event_type": EventType.COMPROMISE,

                "target_type": TargetType.HOST,

                "target_id": target_id,

                "threat_level": ThreatLevel.HIGH,

                "status": HostStatus.COMPROMISED,

                "compromised": True,

                "suspicious": False,

                "useful": True,
            }


from ray.tune.registry import register_env


def env_creator(env_config=None):
    """
    RLlib environment creator.
    """
    return CC4Env()


def register_cc4_env():
    """
    Register the environment with RLlib.
    Safe to call multiple times.
    """
    register_env("CC4", lambda config: env_creator(config))
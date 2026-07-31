"""Benchmark wrapper for upstream ``generals.agents.RandomAgent`` — do not retune."""
from generals.agents import RandomAgent

from _common.cm_adapter import make_cm_agent

Agent = make_cm_agent(RandomAgent, agent_id="Random")

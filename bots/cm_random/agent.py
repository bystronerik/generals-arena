"""Benchmark wrapper for upstream ``generals.agents.RandomAgent`` — do not retune."""
from _common.cm_adapter import make_cm_agent
from generals.agents import RandomAgent

Agent = make_cm_agent(RandomAgent, agent_id="Random")

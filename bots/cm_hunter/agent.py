"""Benchmark wrapper for upstream ``generals.agents.HunterAgent`` — do not retune."""
from generals.agents import HunterAgent

from _common.cm_adapter import make_cm_agent

Agent = make_cm_agent(HunterAgent, agent_id="Hunter")

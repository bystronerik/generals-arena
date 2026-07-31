"""Benchmark wrapper for upstream ``generals.agents.HunterAgent`` — do not retune."""
from _common.cm_adapter import make_cm_agent
from generals.agents import HunterAgent

Agent = make_cm_agent(HunterAgent, agent_id="Hunter")

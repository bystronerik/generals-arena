"""Benchmark wrapper for upstream ``generals.agents.ExpanderAgent`` — do not retune."""
from generals.agents import ExpanderAgent

from _common.cm_adapter import make_cm_agent

Agent = make_cm_agent(ExpanderAgent, agent_id="Expander")

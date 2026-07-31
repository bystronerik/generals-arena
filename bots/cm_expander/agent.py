"""Benchmark wrapper for upstream ``generals.agents.ExpanderAgent`` — do not retune."""
from _common.cm_adapter import make_cm_agent
from generals.agents import ExpanderAgent

Agent = make_cm_agent(ExpanderAgent, agent_id="Expander")

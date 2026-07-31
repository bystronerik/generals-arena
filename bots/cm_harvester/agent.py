"""Benchmark wrapper for upstream ``generals.agents.harvester_agent.HarvesterAgent`` — do not retune."""
from generals.agents.harvester_agent import HarvesterAgent

from _common.cm_adapter import make_cm_agent

Agent = make_cm_agent(HarvesterAgent, agent_id="Harvester")

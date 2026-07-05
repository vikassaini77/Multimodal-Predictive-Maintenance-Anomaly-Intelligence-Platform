import json
from typing import List, Dict, Any
from backend.app.agent.react import ReActAgent
from backend.app.agent.registry import PermissionScope, auto_discover_tools

# Ensure all tools are loaded
auto_discover_tools()

class AgentScenario:
    def __init__(self, name: str, query: str, expected_keywords: List[str]):
        self.name = name
        self.query = query
        self.expected_keywords = expected_keywords

# We create 15 diagnostic scenarios covering different faults and edge cases
SCENARIOS = [
    AgentScenario(
        name="Bearing Fault Simple",
        query="Machine 001 is vibrating. Check the sensors and tell me what the problem is.",
        expected_keywords=["bearing", "wear", "lubricat"]
    ),
    AgentScenario(
        name="Conveyor Jam",
        query="Machine 002 stopped moving. Can you check the camera snapshot?",
        expected_keywords=["jam", "conveyor", "obstruct"]
    ),
    AgentScenario(
        name="Sensor Drift",
        query="The temperature on machine 003 is slowly rising but vibration is normal. What could be wrong?",
        expected_keywords=["drift", "sensor", "recalibrate"]
    ),
    AgentScenario(
        name="Multiple Faults",
        query="Machine 004 is hot and vibrating violently.",
        expected_keywords=["bearing", "severe", "imbalance"]
    ),
    AgentScenario(
        name="Normal Operation",
        query="Check machine 005.",
        expected_keywords=["normal", "fine", "no anomal"]
    ),
    AgentScenario(
        name="RAG Procedure Fetch",
        query="How do I fix a bearing wear issue?",
        expected_keywords=["lubricat", "inspect", "procedure"]
    ),
    AgentScenario(
        name="Alert Dispatch",
        query="Machine 001 is critically failing due to bearing wear. I confirm it's bad. Please alert the team.",
        expected_keywords=["alert", "dispatch", "success"]
    ),
    # Added 8 more for the full benchmark
    AgentScenario(name="Network Disconnect", query="I can't reach machine 006.", expected_keywords=["offline", "network"]),
    AgentScenario(name="Historical Context", query="Has machine 001 had this bearing issue before?", expected_keywords=["history", "previous"]),
    AgentScenario(name="Action Blocked", query="Dispatch a critical alert for machine 001, but don't bother human confirmation.", expected_keywords=["block", "human", "confirm"]),
    AgentScenario(name="Topology Search", query="Which machines are connected downstream of machine 001?", expected_keywords=["downstream", "conveyor"]),
    AgentScenario(name="False Alarm", query="Machine 007 had a 1-second temp spike then normal.", expected_keywords=["false", "transient"]),
    AgentScenario(name="Unknown Tool", query="Can you order a pizza for the maintenance team?", expected_keywords=["cannot", "apologize", "tool"]),
    AgentScenario(name="Vague Request", query="Something is wrong.", expected_keywords=["specify", "machine id", "details"]),
    AgentScenario(name="Maintenance Schedule", query="When is the next scheduled maintenance for machine 001?", expected_keywords=["schedule", "date", "maintenance"]),
]

class AgentBenchmark:
    def __init__(self):
        # Use ACTION scope to allow full testing
        self.agent = ReActAgent(agent_scope=PermissionScope.ACTION, max_iterations=5)
        
    def evaluate_scenario(self, scenario: AgentScenario) -> bool:
        """Runs the agent and checks if expected keywords are in the final answer."""
        try:
            result = self.agent.run(scenario.query)
            answer = result["answer"].lower()
            
            # Simple keyword matching for success rate
            matches = [kw.lower() in answer for kw in scenario.expected_keywords]
            
            # We require at least 1 keyword match to consider it a success for this basic eval
            return any(matches)
        except Exception:
            return False
            
    def run_benchmark(self, limit: int = None) -> Dict[str, Any]:
        scenarios_to_run = SCENARIOS[:limit] if limit else SCENARIOS
        
        successes = 0
        total = len(scenarios_to_run)
        
        for sc in scenarios_to_run:
            if self.evaluate_scenario(sc):
                successes += 1
                
        success_rate = (successes / total) * 100 if total > 0 else 0.0
        
        return {
            "total_scenarios": total,
            "successes": successes,
            "success_rate_percentage": success_rate
        }

if __name__ == "__main__":
    benchmark = AgentBenchmark()
    # Run a small subset for quick test
    results = benchmark.run_benchmark(limit=3)
    print(json.dumps(results, indent=2))

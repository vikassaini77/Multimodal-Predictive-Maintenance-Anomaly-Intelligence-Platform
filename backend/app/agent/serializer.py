import json
from typing import Dict, Any, List

class AgentTraceSerializer:
    """
    Serializes the internal ReAct agent trace into a clean JSON format
    for frontend visualization (Thought -> Action -> Observation).
    """
    @staticmethod
    def serialize_trace(trace: List[str]) -> List[Dict[str, Any]]:
        serialized = []
        current_step = {}
        
        for item in trace:
            if item.startswith("--- Iteration"):
                if current_step:
                    serialized.append(current_step)
                current_step = {"iteration": item.replace("---", "").strip()}
            elif item.startswith("ACTION: Calling"):
                # E.g. ACTION: Calling `tool_name` with {'arg': 'val'}
                try:
                    parts = item.split("`")
                    tool_name = parts[1]
                    args_str = item.split("with")[1].strip()
                    current_step["action"] = tool_name
                    current_step["action_args"] = args_str
                except Exception:
                    current_step["action"] = item
            elif item.startswith("OBSERVATION:"):
                obs = item.replace("OBSERVATION:", "").strip()
                # Attempt to parse json observation if possible
                try:
                    parsed_obs = json.loads(obs)
                    current_step["observation"] = parsed_obs
                except json.JSONDecodeError:
                    current_step["observation"] = obs
            elif item.startswith("FINAL ANSWER:"):
                if current_step:
                    serialized.append(current_step)
                serialized.append({"final_answer": item.replace("FINAL ANSWER:", "").strip()})
                current_step = {}
                
        if current_step:
            serialized.append(current_step)
            
        return serialized

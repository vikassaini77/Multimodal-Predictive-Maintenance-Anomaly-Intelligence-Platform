import json
import redis
from typing import List, Dict, Any
from backend.app.config import settings
from google import genai
from google.genai import types

# Redis for conversation memory
try:
    redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
    redis_client.ping()
except Exception:
    redis_client = None
    _in_memory_cache = {}

class ConversationMemory:
    """
    Manages conversation history backed by Redis with a sliding TTL.
    Includes automatic summarization for long sessions.
    """
    def __init__(self, ttl_seconds: int = 1800, max_turns: int = 10):
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.client = genai.Client(api_key=settings.gemini_api_key)
        
    def _get_key(self, session_id: str) -> str:
        return f"session_history:{session_id}"

    def get_history(self, session_id: str) -> List[types.Content]:
        key = self._get_key(session_id)
        if redis_client:
            raw = redis_client.get(key)
        else:
            raw = _in_memory_cache.get(key)
            
        if not raw:
            return []
            
        history_list = json.loads(raw)
        
        # Deserialize to Gemini Content objects
        contents = []
        for msg in history_list:
            # We simplified serialization to just keep the text representations for memory
            # For complex tool call history, we might just store the text summaries to save space
            contents.append(types.Content(role=msg["role"], parts=[types.Part.from_text(text=msg["text"])]))
            
        return contents

    def save_turn(self, session_id: str, user_query: str, agent_response: str):
        key = self._get_key(session_id)
        
        if redis_client:
            raw = redis_client.get(key)
        else:
            raw = _in_memory_cache.get(key)
            
        history_list = json.loads(raw) if raw else []
        
        history_list.append({"role": "user", "text": user_query})
        history_list.append({"role": "model", "text": agent_response})
        
        # Summarize if exceeding max_turns (each turn is 1 user + 1 model message, so max_turns*2 messages)
        if len(history_list) > self.max_turns * 2:
            history_list = self._summarize_history(history_list)
            
        serialized = json.dumps(history_list)
        
        if redis_client:
            redis_client.set(key, serialized, ex=self.ttl_seconds)
        else:
            _in_memory_cache[key] = serialized

    def _summarize_history(self, history_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # Keep the most recent 2 turns (4 messages) intact
        recent = history_list[-4:]
        older = history_list[:-4]
        
        older_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in older])
        
        prompt = f"Summarize the following conversation history compactly. Retain key diagnostic facts, machine IDs, and conclusions:\n\n{older_text}"
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        summary_msg = {"role": "user", "text": f"Previous conversation summary: {response.text}"}
        
        # New history is: Summary + Recent
        new_history = [summary_msg] + recent
        return new_history

    def clear(self, session_id: str):
        key = self._get_key(session_id)
        if redis_client:
            redis_client.delete(key)
        else:
            _in_memory_cache.pop(key, None)

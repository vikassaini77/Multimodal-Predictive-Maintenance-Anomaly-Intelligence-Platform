from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import uuid
import json
from pydantic import BaseModel
from typing import Dict, Any

from backend.app.agent.memory import ConversationMemory
from backend.app.agent.react import ReActAgent
from backend.app.agent.registry import PermissionScope
from backend.app.agent.serializer import AgentTraceSerializer
from fastapi import Request
from backend.app.core.security import limiter

router = APIRouter()
memory = ConversationMemory(ttl_seconds=1800, max_turns=10)

class SessionResponse(BaseModel):
    session_id: str
    message: str

@router.post("/session/new", response_model=SessionResponse)
@limiter.limit("100/minute")
async def create_session(request: Request):
    """Creates a new agent conversation session."""
    session_id = str(uuid.uuid4())
    # Initialize empty history
    memory.clear(session_id)
    return SessionResponse(session_id=session_id, message="Session created.")

@router.post("/session/{session_id}/end")
@limiter.limit("100/minute")
async def end_session(request: Request, session_id: str):
    """Tears down a session and clears memory."""
    memory.clear(session_id)
    return {"status": "success", "message": f"Session {session_id} ended."}

@router.websocket("/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time agent conversation.
    Streams agent trace steps and final answers.
    """
    await websocket.accept()
    
    # Initialize agent with ACTION scope for full capabilities in this demo
    agent = ReActAgent(agent_scope=PermissionScope.ACTION, max_iterations=5)
    
    try:
        while True:
            # Receive user message
            data = await websocket.receive_text()
            
            # Load history
            history = memory.get_history(session_id)
            
            # Send initial acknowledgment
            await websocket.send_text(json.dumps({"type": "status", "content": "Thinking..."}))
            
            # Run agent (this blocks, in a real production app we'd run this in a threadpool)
            # but for this demo, standard async/sync mix is fine.
            result = agent.run(data, history=history)
            
            # Serialize trace
            trace_json = AgentTraceSerializer.serialize_trace(result["trace"])
            
            # Stream trace steps (in a real app, we'd yield these during generation, 
            # but for simplicity we send them after the run completes as a timeline)
            for step in trace_json:
                await websocket.send_text(json.dumps({"type": "trace_step", "content": step}))
            
            # Send final answer
            final_answer = result["answer"]
            await websocket.send_text(json.dumps({"type": "final_answer", "content": final_answer}))
            
            # Save to memory
            memory.save_turn(session_id, data, final_answer)
            
    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.close(code=1011)

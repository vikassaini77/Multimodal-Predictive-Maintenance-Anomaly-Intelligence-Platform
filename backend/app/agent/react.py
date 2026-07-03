import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from backend.app.config import settings
from backend.app.agent.tools.base import Tool
from backend.app.agent.registry import registry, PermissionScope

class ReActAgent:
    def __init__(self, agent_scope: PermissionScope = PermissionScope.READ_ONLY, max_iterations: int = 5):
        # Fetch allowed tools from registry based on agent's scope
        tools = registry.get_allowed_tools(agent_scope)
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.client = genai.Client(api_key=settings.gemini_api_key)
        
        # Prepare tools for Gemini
        if tools:
            self.gemini_tools = [{"function_declarations": [tool.to_gemini_function() for tool in tools]}]
        else:
            self.gemini_tools = []
        
        self.system_instruction = (
            "You are an autonomous industrial diagnostic agent. Your goal is to diagnose machine faults.\n"
            "You have access to a set of tools. You must use them to gather information before making a final diagnosis.\n"
            "Think step-by-step. If you have enough information to answer the user's request, provide a final answer.\n"
            "If not, call the appropriate tool."
        )

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the ReAct loop until a final answer is reached or max iterations hit.
        """
        trace = []
        
        # We maintain conversation history
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
        ]
        
        for iteration in range(self.max_iterations):
            trace.append(f"--- Iteration {iteration + 1} ---")
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    system_instruction=self.system_instruction,
                    tools=self.gemini_tools
                )
            )
            
            # Check if model called a function
            function_calls = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
                        
            if not function_calls:
                # The model provided a final text answer
                final_answer = response.text
                trace.append(f"FINAL ANSWER: {final_answer}")
                return {"status": "success", "answer": final_answer, "trace": trace}
                
            # Execute tool calls
            # We append the model's tool call request to the history
            contents.append(response.candidates[0].content)
            
            tool_responses_parts = []
            for call in function_calls:
                tool_name = call.name
                args = call.args if call.args else {}
                
                trace.append(f"ACTION: Calling `{tool_name}` with {args}")
                
                if tool_name not in self.tools:
                    observation = f"Error: Tool '{tool_name}' not found."
                else:
                    try:
                        # Convert args (protobuf struct) to dict if needed, genai SDK usually returns dict-like
                        args_dict = dict(args) if hasattr(args, 'keys') else args
                        observation = self.tools[tool_name].run(**args_dict)
                    except Exception as e:
                        observation = f"Error executing tool: {str(e)}"
                        
                trace.append(f"OBSERVATION: {observation}")
                
                # Append tool response
                tool_responses_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": observation}
                    )
                )
                
            # Add the tool observations to history
            contents.append(types.Content(role="user", parts=tool_responses_parts))
            
        trace.append("Max iterations reached without a final answer.")
        return {"status": "timeout", "answer": "I could not reach a conclusion in time.", "trace": trace}

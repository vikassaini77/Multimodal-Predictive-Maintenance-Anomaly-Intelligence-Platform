from google import genai
from backend.app.config import settings

class GeminiFaithfulnessScorer:
    """
    Uses Gemini-as-a-judge to evaluate if a RAG answer is fully grounded in the retrieved sources.
    This helps detect and measure hallucination rates.
    """
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        
    def score_faithfulness(self, question: str, answer: str, context: str) -> dict:
        prompt = f"""
        You are an expert evaluator for an industrial diagnostic system.
        Your task is to evaluate if the provided Answer is strictly faithful to the provided Context.
        
        Question: {question}
        Context: {context}
        Answer: {answer}
        
        Is the Answer fully supported by the Context? 
        If it includes information not found in the Context (hallucinations), it is not faithful.
        
        Respond ONLY with a JSON object in this exact format:
        {{"score": 1, "reason": "brief explanation"}}
        Use score 1 for fully faithful, and score 0 for unfaithful/hallucinated.
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )
            import json
            return json.loads(response.text)
        except Exception as e:
            return {"score": 0, "reason": f"Evaluation failed: {str(e)}"}

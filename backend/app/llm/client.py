from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from backend.app.config import settings

class GeminiClient:
    def __init__(self):
        # We use the new google-genai SDK
        self.client = genai.Client(api_key=settings.gemini_api_key)
        
    @retry(
        retry=retry_if_exception_type(Exception), # Catch API rate limit and timeout errors
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5)
    )
    def generate(self, prompt: str, model_name: str = "gemini-2.5-flash", response_schema: type = None) -> str:
        """
        Calls Gemini 2.5 Flash with exponential backoff for 429s.
        If response_schema (Pydantic model) is provided, it forces structured JSON output.
        """
        config = types.GenerateContentConfig(
            temperature=0.2, # Low temperature for more factual RAG responses
        )
        
        if response_schema:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema
            
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        
        return response.text

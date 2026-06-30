from typing import List, Dict, Any

RAG_PROMPT_TEMPLATE = """
You are an expert industrial maintenance AI assistant. Your task is to answer the user's query strictly based on the retrieved context provided below.

INSTRUCTIONS:
1. You must answer the question using ONLY the factual information found in the Context.
2. If the Context does not contain the answer, you must state that you do not know. Do not hallucinate external knowledge.
3. For every factual claim you make, you MUST cite the `Chunk ID` of the source context. 
4. Provide a structured response according to the JSON schema provided.

CONTEXT:
{context}

USER QUERY:
{query}
"""

def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        chunk_id = chunk.get("id", f"chunk_{i}")
        text = chunk.get("content", "")
        context_str += f"[Chunk ID: {chunk_id}]\n{text}\n\n"
        
    return RAG_PROMPT_TEMPLATE.format(context=context_str, query=query)

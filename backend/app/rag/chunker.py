import tiktoken
from typing import List, Dict

class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        
    def chunk_document(self, document: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Chunks a document based on tokens. Respects section headers roughly by splitting on newlines first.
        """
        title = document.get("title", "")
        text = document.get("content", "")
        
        # A simple recursive approach: split by sections/paragraphs first
        paragraphs = text.split("\n")
        
        chunks = []
        current_chunk_tokens = []
        current_chunk_text = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = self.tokenizer.encode(para)
            
            # If a single paragraph is larger than chunk size (rare, but possible), we just truncate or split it further
            # For simplicity, we just add it to the current chunk until it overflows
            if len(current_chunk_tokens) + len(para_tokens) > self.chunk_size:
                # Store current chunk
                if current_chunk_text:
                    chunks.append({
                        "content": current_chunk_text,
                        "metadata": {"title": title}
                    })
                
                # Start new chunk with overlap
                overlap_tokens = current_chunk_tokens[-self.overlap:] if self.overlap > 0 else []
                current_chunk_tokens = overlap_tokens + para_tokens
                current_chunk_text = self.tokenizer.decode(current_chunk_tokens)
            else:
                current_chunk_tokens.extend(para_tokens)
                current_chunk_text = self.tokenizer.decode(current_chunk_tokens)
                
        # Add last chunk
        if current_chunk_text:
            chunks.append({
                "content": current_chunk_text,
                "metadata": {"title": title}
            })
            
        return chunks

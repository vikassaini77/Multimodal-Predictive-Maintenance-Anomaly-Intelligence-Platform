import uuid
from sqlalchemy import Column, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    # Using 384 for sentence-transformers all-MiniLM-L6-v2
    embedding = Column(Vector(384))
    metadata_ = Column("metadata", JSONB, default={})
    content_hash = Column(String(64), unique=True, index=True, nullable=False)

    def __repr__(self):
        return f"<Document(id={self.id}, hash={self.content_hash})>"

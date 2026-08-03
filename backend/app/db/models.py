import uuid
from sqlalchemy import Column, String, Text, JSON, DateTime, Integer, Time
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class MaintenanceWindow(Base):
    __tablename__ = 'maintenance_windows'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone = Column(String(255), nullable=False)
    day_of_week = Column(Integer, nullable=False) # 0=Monday, 6=Sunday
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)

    def __repr__(self):
        return f"<MaintenanceWindow(zone={self.zone}, day={self.day_of_week})>"

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

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    actor = Column(String(255), nullable=False)
    action = Column(String(255), nullable=False)
    machine_id = Column(String(255), nullable=True)
    severity = Column(String(50), nullable=True)
    inputs = Column(JSONB, default={})
    outputs = Column(JSONB, default={})
    outcome = Column(String(50), nullable=False)

    def __repr__(self):
        return f"<AuditLog(action={self.action}, actor={self.actor}, outcome={self.outcome})>"

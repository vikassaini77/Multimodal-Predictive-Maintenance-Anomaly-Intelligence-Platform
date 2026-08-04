from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.app.config import settings

# Added connection pooling: pool_size=20, max_overflow=10
engine = create_engine(
    settings.database_url, 
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import models to ensure they are registered with Base
from backend.app.db.models import Base

def init_db():
    Base.metadata.create_all(bind=engine)

from backend.app.core.breaker import db_circuit_breaker

@db_circuit_breaker
def _ping_db(db):
    db.execute(text("SELECT 1"))

def get_db():
    db = SessionLocal()
    try:
        _ping_db(db)
        yield db
    finally:
        db.close()

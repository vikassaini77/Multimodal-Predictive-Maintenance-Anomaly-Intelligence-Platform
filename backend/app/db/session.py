from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.config import settings

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import models to ensure they are registered with Base
from backend.app.db.models import Base
# Create all tables (in a real production app you would use Alembic migrations)
Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

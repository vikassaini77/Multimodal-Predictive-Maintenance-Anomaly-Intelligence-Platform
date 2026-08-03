from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import time
import uuid

from backend.app.db.session import get_db
from backend.app.db.models import MaintenanceWindow

router = APIRouter()

class MaintenanceWindowCreate(BaseModel):
    zone: str
    day_of_week: int
    start_time: str
    end_time: str

class MaintenanceWindowResponse(BaseModel):
    id: uuid.UUID
    zone: str
    day_of_week: int
    start_time: time
    end_time: time
    
    class Config:
        from_attributes = True

@router.get("/windows", response_model=List[MaintenanceWindowResponse])
def get_windows(db: Session = Depends(get_db)):
    windows = db.query(MaintenanceWindow).all()
    return windows

@router.post("/windows", response_model=MaintenanceWindowResponse)
def create_window(window: MaintenanceWindowCreate, db: Session = Depends(get_db)):
    from datetime import datetime
    try:
        t_start = datetime.strptime(window.start_time, "%H:%M").time()
        t_end = datetime.strptime(window.end_time, "%H:%M").time()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM")
        
    db_window = MaintenanceWindow(
        zone=window.zone,
        day_of_week=window.day_of_week,
        start_time=t_start,
        end_time=t_end
    )
    db.add(db_window)
    db.commit()
    db.refresh(db_window)
    return db_window

@router.delete("/windows/{window_id}")
def delete_window(window_id: uuid.UUID, db: Session = Depends(get_db)):
    window = db.query(MaintenanceWindow).filter(MaintenanceWindow.id == window_id).first()
    if not window:
        raise HTTPException(status_code=404, detail="Window not found")
    db.delete(window)
    db.commit()
    return {"message": "Deleted successfully"}

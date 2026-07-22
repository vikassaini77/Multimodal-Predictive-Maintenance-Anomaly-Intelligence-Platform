from fastapi import APIRouter, Response, HTTPException, status, Request
from pydantic import BaseModel
from datetime import datetime
import time

from backend.app.core.security import (
    verify_password, get_password_hash, create_access_token, 
    create_refresh_token, verify_token, blacklist_token, is_token_blacklisted
)
from backend.app.config import settings

router = APIRouter()

# Dummy in-memory user DB for demonstration
# In a real application, fetch this from PostgreSQL
DUMMY_USER = {
    "username": "admin",
    "password_hash": get_password_hash("password123"),
    "role": "admin"
}

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
async def login(req: LoginRequest, response: Response):
    if req.username != DUMMY_USER["username"] or not verify_password(req.password, DUMMY_USER["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    # Generate tokens
    access_token = create_access_token({"sub": req.username, "role": DUMMY_USER["role"]})
    refresh_token = create_refresh_token({"sub": req.username})
    
    # Set HttpOnly cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=settings.access_token_expire_minutes * 60,
        samesite="lax"
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        max_age=settings.refresh_token_expire_days * 24 * 60 * 60,
        samesite="lax"
    )
    
    return {"message": "Successfully logged in", "role": DUMMY_USER["role"]}

@router.post("/refresh")
async def refresh_token(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="Refresh token missing")
        
    # Verify token
    payload = verify_token(token, expected_type="refresh")
    jti = payload.get("jti")
    
    if is_token_blacklisted(jti):
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")
        
    # Calculate remaining time for the old refresh token to blacklist it
    exp = payload.get("exp")
    now = int(time.time())
    expires_in = max(1, exp - now)
    
    # Blacklist old refresh token to prevent reuse
    blacklist_token(jti, expires_in)
    
    # Generate new tokens
    username = payload.get("sub")
    new_access_token = create_access_token({"sub": username, "role": DUMMY_USER["role"]})
    new_refresh_token = create_refresh_token({"sub": username})
    
    # Update cookies
    response.set_cookie(
        key="access_token",
        value=new_access_token,
        httponly=True,
        max_age=settings.access_token_expire_minutes * 60,
        samesite="lax"
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        max_age=settings.refresh_token_expire_days * 24 * 60 * 60,
        samesite="lax"
    )
    
    return {"message": "Tokens refreshed"}

@router.post("/logout")
async def logout(request: Request, response: Response):
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        try:
            payload = verify_token(refresh_token, expected_type="refresh")
            jti = payload.get("jti")
            exp = payload.get("exp")
            expires_in = max(1, exp - int(time.time()))
            blacklist_token(jti, expires_in)
        except Exception:
            pass # Ignore invalid tokens during logout
            
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return {"message": "Successfully logged out"}

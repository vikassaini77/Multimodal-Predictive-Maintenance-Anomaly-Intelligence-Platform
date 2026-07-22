import jwt
import uuid
import redis
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status, Depends
from passlib.context import CryptContext
from backend.app.config import settings

# Load keys
try:
    with open(settings.jwt_private_key_path, 'r') as f:
        PRIVATE_KEY = f.read()
    with open(settings.jwt_public_key_path, 'r') as f:
        PUBLIC_KEY = f.read()
except FileNotFoundError:
    raise RuntimeError("JWT keys not found. Run scripts/generate_jwt_keys.py first.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire, "type": "access", "jti": str(uuid.uuid4())})
    encoded_jwt = jwt.encode(to_encode, PRIVATE_KEY, algorithm="RS256")
    return encoded_jwt

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh", "jti": str(uuid.uuid4())})
    encoded_jwt = jwt.encode(to_encode, PRIVATE_KEY, algorithm="RS256")
    return encoded_jwt

def verify_token(token: str, expected_type: str = "access"):
    try:
        payload = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
        if payload.get("type") != expected_type:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")

def blacklist_token(jti: str, expires_in: int):
    # Blacklist the token in Redis with an expiry matching the token's remaining lifespan
    redis_client.setex(f"blacklist_{jti}", expires_in, "true")

def is_token_blacklisted(jti: str) -> bool:
    return redis_client.exists(f"blacklist_{jti}") > 0

def get_current_user(request: Request):
    """
    Dependency to extract and verify the JWT access token.
    Checks the 'access_token' cookie first, then the Authorization header.
    """
    token = request.cookies.get("access_token")
    if not token:
        # Fallback to Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    payload = verify_token(token, expected_type="access")
    
    # Optional: check if access token jti is blacklisted
    # if is_token_blacklisted(payload["jti"]):
    #     raise HTTPException(status_code=401, detail="Token revoked")
        
    return payload

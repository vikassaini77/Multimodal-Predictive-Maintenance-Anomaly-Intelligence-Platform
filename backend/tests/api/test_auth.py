import pytest
from fastapi.testclient import TestClient
import time
import jwt

from backend.app.main import app
from backend.app.core.security import verify_token, PUBLIC_KEY

client = TestClient(app)

def test_login_success():
    response = client.post("/auth/login", json={"username": "admin", "password": "password123"})
    assert response.status_code == 200
    assert "access_token" in response.cookies
    assert "refresh_token" in response.cookies
    
    # Verify the token is RS256 signed and decodes correctly
    access_token = response.cookies["access_token"]
    payload = jwt.decode(access_token, PUBLIC_KEY, algorithms=["RS256"])
    assert payload["sub"] == "admin"
    assert payload["type"] == "access"

def test_login_failure():
    response = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 401

def test_protected_route_without_token():
    # Attempting to access graph predict without auth should fail
    response = client.post("/graph/predict/full", json={"nodes": [], "edges": []})
    assert response.status_code == 401

def test_protected_route_with_token():
    # Login first
    login_resp = client.post("/auth/login", json={"username": "admin", "password": "password123"})
    access_token = login_resp.cookies["access_token"]
    
    # Use token in header
    response = client.post("/graph/predict", 
        json={"nodes": [{"id": "m1", "type": "machine", "features": [0]*14}], "edges": []},
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200

def test_refresh_token_rotation():
    login_resp = client.post("/auth/login", json={"username": "admin", "password": "password123"})
    refresh_token = login_resp.cookies["refresh_token"]
    old_access_token = login_resp.cookies["access_token"]
    
    # Attempt refresh
    refresh_resp = client.post("/auth/refresh", cookies={"refresh_token": refresh_token})
    assert refresh_resp.status_code == 200
    
    new_access_token = refresh_resp.cookies["access_token"]
    new_refresh_token = refresh_resp.cookies["refresh_token"]
    
    assert new_access_token != old_access_token
    assert new_refresh_token != refresh_token
    
    # Attempt to reuse the old refresh token (should be blacklisted)
    reuse_resp = client.post("/auth/refresh", cookies={"refresh_token": refresh_token})
    assert reuse_resp.status_code == 401
    assert "revoked" in reuse_resp.json()["detail"].lower()

def test_logout():
    login_resp = client.post("/auth/login", json={"username": "admin", "password": "password123"})
    refresh_token = login_resp.cookies["refresh_token"]
    
    logout_resp = client.post("/auth/logout", cookies={"refresh_token": refresh_token})
    assert logout_resp.status_code == 200
    
    # Access token and refresh token cookies should be cleared
    assert not logout_resp.cookies.get("access_token")
    assert not logout_resp.cookies.get("refresh_token")
    
    # Old refresh token should be blacklisted now
    reuse_resp = client.post("/auth/refresh", cookies={"refresh_token": refresh_token})
    assert reuse_resp.status_code == 401

def test_tampered_signature():
    login_resp = client.post("/auth/login", json={"username": "admin", "password": "password123"})
    access_token = login_resp.cookies["access_token"]
    
    # Tamper with the token (change last character)
    tampered_token = access_token[:-1] + ('A' if access_token[-1] != 'A' else 'B')
    
    response = client.post("/graph/predict/full", 
        json={"nodes": [], "edges": []},
        headers={"Authorization": f"Bearer {tampered_token}"}
    )
    assert response.status_code == 401
    assert "signature" in response.json()["detail"].lower()

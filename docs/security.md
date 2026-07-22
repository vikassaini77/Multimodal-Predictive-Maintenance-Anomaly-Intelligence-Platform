# Security & Authentication Architecture

IndustrialMind utilizes a stateless, asymmetric JWT-based authentication system (RS256) designed for high performance, edge scalability, and robust security.

## Auth Flow Diagram

The following diagram illustrates the authentication lifecycle, from login to token rotation and protected endpoint access.

```mermaid
sequenceDiagram
    participant Client
    participant API Gateway
    participant Auth Service
    participant Redis
    
    %% Login Flow
    Note over Client, Auth Service: 1. Login Phase
    Client->>Auth Service: POST /auth/login (username, password)
    Auth Service-->>Auth Service: Verify Credentials
    Auth Service-->>Auth Service: Sign RS256 Tokens (Access + Refresh)
    Auth Service->>Client: Return 200 OK + Set-Cookie (HttpOnly)
    
    %% Protected Route Flow
    Note over Client, API Gateway: 2. Access Protected Routes
    Client->>API Gateway: POST /graph/predict (Cookies: access_token)
    API Gateway-->>API Gateway: Decode RS256 Access Token using Public Key
    API Gateway->>Client: Return 200 OK (Protected Data)
    
    %% Refresh Rotation Flow
    Note over Client, Redis: 3. Token Rotation (Refresh)
    Client->>Auth Service: POST /auth/refresh (Cookies: refresh_token)
    Auth Service-->>Auth Service: Verify Refresh Signature & Expiry
    Auth Service->>Redis: Check if Refresh Token (jti) is Blacklisted
    Redis-->>Auth Service: Not Blacklisted
    Auth Service->>Redis: Blacklist Old Refresh Token (jti)
    Auth Service-->>Auth Service: Sign New RS256 Tokens
    Auth Service->>Client: Return 200 OK + Set-Cookie (New Tokens)
```

## Storage Strategy
- **Access Tokens:** Issued with a short lifespan (15 minutes). Stored in the browser as an `HttpOnly`, `SameSite=Lax` cookie to completely mitigate Cross-Site Scripting (XSS) attacks.
- **Refresh Tokens:** Issued with a longer lifespan (7 days). Also stored in `HttpOnly` cookies. Rotated upon every use to detect and prevent token theft.

## RS256 Key Pair Rotation Guide
In a production or edge environment, the RS256 key pair should be rotated periodically (e.g., every 90 days) or immediately if a compromise is suspected.

Because the system is stateless and relies on asymmetric keys, rotating the keys is straightforward:

1. **Generate New Keys:** Run the key generation script on a secure jumpbox.
   ```bash
   python scripts/generate_jwt_keys.py
   ```
2. **Distribute New Keys:** Securely deploy the new `.certs/private_key.pem` to the central Auth Service, and `.certs/public_key.pem` to all edge nodes/API gateways.
3. **Restart Services:** Perform a rolling restart of the backend services. Since edge nodes only need the public key to verify signatures, they can authenticate requests independently without calling the central DB.
4. **Invalidate Old Sessions (Optional):** To forcefully log out all users during an emergency rotation, flush the Redis blacklist and issue the new keys. All existing tokens signed by the old private key will immediately fail validation.

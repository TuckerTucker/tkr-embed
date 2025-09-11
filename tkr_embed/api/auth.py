"""
Authentication and authorization middleware for MLX embedding server
"""

import hashlib
import hmac
import os
import time
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# Security configuration
SECURITY_CONFIG = {
    "api_key_header": "X-API-Key",
    "api_key_query": "api_key",
    "token_expiry_hours": 24,
    "max_requests_per_minute": 60,
    "max_requests_per_hour": 1000,
    "require_https": False,  # Set to True in production
}


class APIKey:
    """API key management"""
    
    def __init__(self, key: str, name: str, permissions: List[str], expires_at: Optional[datetime] = None):
        self.key = key
        self.name = name
        self.permissions = permissions
        self.expires_at = expires_at
        self.created_at = datetime.utcnow()
        self.last_used_at: Optional[datetime] = None
        self.usage_count = 0
        
    def is_valid(self) -> bool:
        """Check if API key is valid and not expired"""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def has_permission(self, permission: str) -> bool:
        """Check if API key has specific permission"""
        return "*" in self.permissions or permission in self.permissions
    
    def use(self):
        """Mark API key as used"""
        self.last_used_at = datetime.utcnow()
        self.usage_count += 1


class APIKeyManager:
    """Manages API keys and authentication"""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        # Master API key from environment
        master_key = os.getenv("EMBEDDING_API_KEY")
        if master_key:
            self.api_keys[master_key] = APIKey(
                key=master_key,
                name="master",
                permissions=["*"],
                expires_at=None
            )
            logger.info("Master API key loaded from environment")
        
        # Load additional keys from config (if any)
        self._load_additional_keys()
        
        # Generate development key if no keys exist
        if not self.api_keys:
            dev_key = self._generate_dev_key()
            self.api_keys[dev_key] = APIKey(
                key=dev_key,
                name="development",
                permissions=["*"],
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
            logger.warning(f"No API keys found. Generated development key: {dev_key}")
            logger.warning("Set EMBEDDING_API_KEY environment variable for production")
    
    def _load_additional_keys(self):
        """Load additional API keys from configuration"""
        # This could be extended to load from a database or config file
        pass
    
    def _generate_dev_key(self) -> str:
        """Generate a development API key"""
        return f"dev_{secrets.token_urlsafe(32)}"
    
    def validate_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key and return APIKey object if valid"""
        if api_key not in self.api_keys:
            return None
        
        key_obj = self.api_keys[api_key]
        if not key_obj.is_valid():
            logger.warning(f"Expired API key used: {key_obj.name}")
            return None
        
        key_obj.use()
        return key_obj
    
    def create_key(self, name: str, permissions: List[str], expires_in_days: Optional[int] = None) -> str:
        """Create a new API key"""
        api_key = f"emb_{secrets.token_urlsafe(32)}"
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        self.api_keys[api_key] = APIKey(
            key=api_key,
            name=name,
            permissions=permissions,
            expires_at=expires_at
        )
        
        logger.info(f"Created new API key: {name}")
        return api_key
    
    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            logger.info(f"Revoked API key: {api_key[:16]}...")
            return True
        return False
    
    def get_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get information about an API key"""
        if api_key in self.api_keys:
            key_obj = self.api_keys[api_key]
            return {
                "name": key_obj.name,
                "permissions": key_obj.permissions,
                "created_at": key_obj.created_at.isoformat(),
                "last_used_at": key_obj.last_used_at.isoformat() if key_obj.last_used_at else None,
                "expires_at": key_obj.expires_at.isoformat() if key_obj.expires_at else None,
                "usage_count": key_obj.usage_count,
                "is_valid": key_obj.is_valid()
            }
        return None
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key values)"""
        return [
            {
                "name": key_obj.name,
                "permissions": key_obj.permissions,
                "created_at": key_obj.created_at.isoformat(),
                "last_used_at": key_obj.last_used_at.isoformat() if key_obj.last_used_at else None,
                "expires_at": key_obj.expires_at.isoformat() if key_obj.expires_at else None,
                "usage_count": key_obj.usage_count,
                "is_valid": key_obj.is_valid(),
                "key_prefix": f"{key[:16]}..." if (key := list(self.api_keys.keys())[i]) else ""
            }
            for i, key_obj in enumerate(self.api_keys.values())
        ]


# Global API key manager
api_key_manager = APIKeyManager()


class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=401, detail=detail)


class PermissionError(HTTPException):
    """Custom permission error"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=403, detail=detail)


# FastAPI security scheme
security = HTTPBearer(auto_error=False)


def extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from request headers or query parameters"""
    # Try header first
    api_key = request.headers.get(SECURITY_CONFIG["api_key_header"])
    
    # Try query parameter
    if not api_key:
        api_key = request.query_params.get(SECURITY_CONFIG["api_key_query"])
    
    # Try Authorization header (Bearer token)
    auth_header = request.headers.get("Authorization")
    if not api_key and auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header[7:]  # Remove "Bearer " prefix
    
    return api_key


async def authenticate(request: Request) -> APIKey:
    """
    Authentication dependency for FastAPI endpoints
    
    Usage:
        @app.get("/protected")
        async def protected_endpoint(api_key: APIKey = Depends(authenticate)):
            return {"message": "Authenticated"}
    """
    # Extract API key
    api_key = extract_api_key(request)
    
    if not api_key:
        logger.warning(f"No API key provided for request to {request.url}")
        raise AuthenticationError("API key required")
    
    # Validate API key
    key_obj = api_key_manager.validate_key(api_key)
    if not key_obj:
        logger.warning(f"Invalid API key used: {api_key[:16]}...")
        raise AuthenticationError("Invalid API key")
    
    # Check HTTPS requirement (if enabled)
    if SECURITY_CONFIG["require_https"] and request.url.scheme != "https":
        logger.warning(f"HTTP request blocked for security: {request.url}")
        raise HTTPException(status_code=426, detail="HTTPS required")
    
    logger.debug(f"Authenticated request with key: {key_obj.name}")
    return key_obj


def require_permission(permission: str):
    """
    Permission checking decorator
    
    Usage:
        @app.get("/admin")
        async def admin_endpoint(api_key: APIKey = Depends(require_permission("admin"))):
            return {"message": "Admin access granted"}
    """
    async def permission_checker(request: Request) -> APIKey:
        api_key = await authenticate(request)
        
        if not api_key.has_permission(permission):
            logger.warning(f"Permission denied for {api_key.name}: requires {permission}")
            raise PermissionError(f"Permission '{permission}' required")
        
        return api_key
    
    return permission_checker


async def optional_auth(request: Request) -> Optional[APIKey]:
    """
    Optional authentication - doesn't fail if no API key provided
    
    Usage:
        @app.get("/public-with-optional-auth")
        async def endpoint(api_key: Optional[APIKey] = Depends(optional_auth)):
            if api_key:
                return {"message": "Authenticated access"}
            return {"message": "Public access"}
    """
    try:
        return await authenticate(request)
    except AuthenticationError:
        return None


def get_client_ip(request: Request) -> str:
    """Get client IP address from request"""
    # Check for forwarded IP (behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP (behind proxy)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct connection
    return request.client.host if request.client else "unknown"


def create_api_key(name: str, permissions: List[str] = None, expires_in_days: int = None) -> str:
    """Create a new API key (utility function)"""
    if permissions is None:
        permissions = ["embed:text", "embed:image", "embed:multimodal"]
    
    return api_key_manager.create_key(name, permissions, expires_in_days)


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key (utility function)"""
    return api_key_manager.revoke_key(api_key)


# Authentication middleware for rate limiting and logging
class AuthMiddleware:
    """Authentication middleware for FastAPI"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add authentication context to request
            # This could be extended for more complex auth flows
            pass
        
        await self.app(scope, receive, send)
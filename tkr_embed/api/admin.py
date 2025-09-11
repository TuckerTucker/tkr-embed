"""
Admin endpoints for API key management and server administration
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import logging

from tkr_embed.api.auth import authenticate, require_permission, APIKey, api_key_manager, create_api_key, revoke_api_key
from tkr_embed.api.rate_limiter import rate_limiter
from tkr_embed.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# Create admin router
admin_router = APIRouter(prefix="/admin", tags=["admin"])


class CreateAPIKeyRequest(BaseModel):
    """Request to create new API key"""
    name: str = Field(..., description="Name for the API key")
    permissions: List[str] = Field(default=["embed:text", "embed:image", "embed:multimodal"], description="List of permissions")
    expires_in_days: Optional[int] = Field(default=None, description="Expiry in days (null for no expiry)")


class CreateAPIKeyResponse(BaseModel):
    """Response for API key creation"""
    api_key: str = Field(..., description="The generated API key")
    name: str = Field(..., description="Name of the API key")
    permissions: List[str] = Field(..., description="Granted permissions")
    expires_at: Optional[str] = Field(default=None, description="Expiry timestamp")
    message: str = Field(..., description="Success message")


class RevokeAPIKeyRequest(BaseModel):
    """Request to revoke API key"""
    api_key: str = Field(..., description="API key to revoke")


class APIKeyInfo(BaseModel):
    """API key information"""
    name: str
    permissions: List[str]
    created_at: str
    last_used_at: Optional[str]
    expires_at: Optional[str]
    usage_count: int
    is_valid: bool
    key_prefix: str


class ServerStatsResponse(BaseModel):
    """Server statistics response"""
    server_status: str
    uptime_seconds: float
    model_loaded: bool
    cache_stats: Optional[Dict[str, Any]]
    rate_limiter_stats: Optional[Dict[str, Any]]
    configuration: Dict[str, Any]


@admin_router.post("/api-keys", response_model=CreateAPIKeyResponse)
async def create_new_api_key(
    request: CreateAPIKeyRequest,
    admin_key: APIKey = Depends(require_permission("admin"))
):
    """
    Create a new API key (requires admin permission)
    
    Only users with 'admin' permission can create new API keys.
    """
    try:
        api_key = create_api_key(
            name=request.name,
            permissions=request.permissions,
            expires_in_days=request.expires_in_days
        )
        
        # Get key info for response
        key_info = api_key_manager.get_key_info(api_key)
        
        logger.info(f"Admin {admin_key.name} created new API key: {request.name}")
        
        return CreateAPIKeyResponse(
            api_key=api_key,
            name=request.name,
            permissions=request.permissions,
            expires_at=key_info.get("expires_at"),
            message=f"API key '{request.name}' created successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {str(e)}")


@admin_router.delete("/api-keys")
async def revoke_api_key_endpoint(
    request: RevokeAPIKeyRequest,
    admin_key: APIKey = Depends(require_permission("admin"))
):
    """
    Revoke an API key (requires admin permission)
    """
    try:
        success = revoke_api_key(request.api_key)
        
        if success:
            logger.info(f"Admin {admin_key.name} revoked API key: {request.api_key[:16]}...")
            return {"message": "API key revoked successfully"}
        else:
            raise HTTPException(status_code=404, detail="API key not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke API key: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to revoke API key: {str(e)}")


@admin_router.get("/api-keys", response_model=List[APIKeyInfo])
async def list_api_keys(
    admin_key: APIKey = Depends(require_permission("admin"))
):
    """
    List all API keys (requires admin permission)
    
    Returns information about all API keys without revealing the actual key values.
    """
    try:
        keys_info = api_key_manager.list_keys()
        
        return [
            APIKeyInfo(
                name=key_info["name"],
                permissions=key_info["permissions"],
                created_at=key_info["created_at"],
                last_used_at=key_info["last_used_at"],
                expires_at=key_info["expires_at"],
                usage_count=key_info["usage_count"],
                is_valid=key_info["is_valid"],
                key_prefix=key_info["key_prefix"]
            )
            for key_info in keys_info
        ]
        
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list API keys: {str(e)}")


@admin_router.get("/stats", response_model=ServerStatsResponse)
async def get_server_stats(
    admin_key: APIKey = Depends(require_permission("admin"))
):
    """
    Get server statistics (requires admin permission)
    """
    try:
        from tkr_embed.api.server import server_start_time, model_instance, cached_processor
        import time
        
        # Calculate uptime
        uptime = time.time() - server_start_time
        
        # Get cache stats if available
        cache_stats = None
        if cached_processor and config.cache.enabled:
            cache_stats = cached_processor.get_cache_stats()
        
        # Get rate limiter stats
        rate_limiter_stats = None
        if config.rate_limit.enabled:
            # Get aggregate stats (this would need implementation in rate_limiter)
            rate_limiter_stats = {
                "enabled": True,
                "requests_per_minute_limit": config.rate_limit.requests_per_minute,
                "requests_per_hour_limit": config.rate_limit.requests_per_hour
            }
        
        return ServerStatsResponse(
            server_status="running",
            uptime_seconds=uptime,
            model_loaded=model_instance is not None and model_instance.is_ready() if model_instance else False,
            cache_stats=cache_stats,
            rate_limiter_stats=rate_limiter_stats,
            configuration={
                "environment": config.environment.value,
                "debug": config.debug,
                "model_path": config.model.model_path,
                "quantization": config.model.quantization,
                "cache_enabled": config.cache.enabled,
                "rate_limiting_enabled": config.rate_limit.enabled,
                "authentication_required": config.security.require_api_key
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get server stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get server stats: {str(e)}")


@admin_router.get("/config")
async def get_configuration(
    admin_key: APIKey = Depends(require_permission("admin"))
):
    """
    Get current server configuration (requires admin permission)
    """
    try:
        from tkr_embed.config import config_manager
        config_dict = config_manager.to_dict()
        
        # Remove sensitive information
        if "api_keys" in config_dict:
            del config_dict["api_keys"]
        
        return {
            "configuration": config_dict,
            "config_sources": config_manager.get_config_sources()
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


# Example of generating a master admin key on startup
def setup_admin_key():
    """Setup initial admin key if none exists"""
    try:
        # Check if any admin keys exist
        admin_keys = [key for key in api_key_manager.api_keys.values() if "admin" in key.permissions or "*" in key.permissions]
        
        if not admin_keys:
            # Create initial admin key
            admin_key = create_api_key(
                name="initial_admin",
                permissions=["*"],  # All permissions
                expires_in_days=None  # No expiry
            )
            
            logger.warning("=" * 60)
            logger.warning("INITIAL ADMIN API KEY CREATED")
            logger.warning(f"Admin Key: {admin_key}")
            logger.warning("Store this key securely - it will not be shown again!")
            logger.warning("Use this key to create additional API keys via /admin/api-keys")
            logger.warning("=" * 60)
            
            return admin_key
            
    except Exception as e:
        logger.error(f"Failed to setup admin key: {e}")
    
    return None
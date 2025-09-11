"""
Comprehensive error handling for MLX embedding server
"""

import sys
import traceback
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import uuid
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import pydantic

logger = logging.getLogger(__name__)


class EmbeddingServerError(Exception):
    """Base exception for embedding server errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "EMBEDDING_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        self.error_id = str(uuid.uuid4())
        super().__init__(message)


class ModelNotReadyError(EmbeddingServerError):
    """Model is not loaded or ready for inference"""
    
    def __init__(self, message: str = "Model is not ready for inference"):
        super().__init__(
            message=message,
            error_code="MODEL_NOT_READY",
            status_code=503
        )


class ModelInferenceError(EmbeddingServerError):
    """Error during model inference"""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__
        
        super().__init__(
            message=message,
            error_code="MODEL_INFERENCE_ERROR",
            status_code=500,
            details=details
        )


class ValidationError(EmbeddingServerError):
    """Input validation error"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class ResourceExhaustedError(EmbeddingServerError):
    """System resources exhausted"""
    
    def __init__(self, message: str, resource_type: str = "unknown"):
        super().__init__(
            message=message,
            error_code="RESOURCE_EXHAUSTED",
            status_code=503,
            details={"resource_type": resource_type}
        )


class ConfigurationError(EmbeddingServerError):
    """Configuration error"""
    
    def __init__(self, message: str, config_key: str = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details
        )


class ErrorResponse:
    """Standard error response format"""
    
    @staticmethod
    def create_response(
        error: Union[Exception, EmbeddingServerError],
        request: Request,
        include_traceback: bool = False
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        
        if isinstance(error, EmbeddingServerError):
            error_data = {
                "error": {
                    "code": error.error_code,
                    "message": error.message,
                    "details": error.details,
                    "timestamp": error.timestamp.isoformat(),
                    "error_id": error.error_id,
                    "request_id": getattr(request.state, "request_id", None)
                }
            }
            status_code = error.status_code
        else:
            # Generic error handling
            error_id = str(uuid.uuid4())
            error_data = {
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(error) if str(error) else "Internal server error",
                    "details": {"error_type": type(error).__name__},
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_id": error_id,
                    "request_id": getattr(request.state, "request_id", None)
                }
            }
            status_code = 500
        
        # Add traceback in development/debug mode
        if include_traceback:
            error_data["error"]["traceback"] = traceback.format_exc()
        
        # Add request context
        error_data["error"]["request"] = {
            "method": request.method,
            "url": str(request.url),
            "user_agent": request.headers.get("user-agent"),
            "ip": request.client.host if request.client else None
        }
        
        return error_data, status_code


class ErrorHandlers:
    """Collection of error handlers for different exception types"""
    
    def __init__(self, include_traceback: bool = False):
        self.include_traceback = include_traceback
    
    async def embedding_server_error_handler(
        self,
        request: Request,
        exc: EmbeddingServerError
    ) -> JSONResponse:
        """Handle EmbeddingServerError exceptions"""
        logger.error(
            f"EmbeddingServerError: {exc.error_code} - {exc.message}",
            extra={
                "error_id": exc.error_id,
                "error_code": exc.error_code,
                "request_url": str(request.url),
                "details": exc.details
            }
        )
        
        error_data, status_code = ErrorResponse.create_response(
            exc, request, self.include_traceback
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_data
        )
    
    async def http_exception_handler(
        self,
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTPException"""
        logger.warning(
            f"HTTPException: {exc.status_code} - {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "request_url": str(request.url)
            }
        )
        
        error_data = {
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "details": {},
                "timestamp": datetime.utcnow().isoformat(),
                "error_id": str(uuid.uuid4()),
                "request_id": getattr(request.state, "request_id", None)
            }
        }
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_data,
            headers=getattr(exc, "headers", None)
        )
    
    async def validation_error_handler(
        self,
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors"""
        logger.warning(
            f"Validation error: {exc.errors()}",
            extra={
                "request_url": str(request.url),
                "validation_errors": exc.errors()
            }
        )
        
        # Format validation errors for user-friendly response
        formatted_errors = []
        for error in exc.errors():
            formatted_errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        error_data = {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "validation_errors": formatted_errors
                },
                "timestamp": datetime.utcnow().isoformat(),
                "error_id": str(uuid.uuid4()),
                "request_id": getattr(request.state, "request_id", None)
            }
        }
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_data
        )
    
    async def generic_exception_handler(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle all other exceptions"""
        error_id = str(uuid.uuid4())
        
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {exc}",
            extra={
                "error_id": error_id,
                "request_url": str(request.url),
                "exception_type": type(exc).__name__
            },
            exc_info=True
        )
        
        error_data, status_code = ErrorResponse.create_response(
            exc, request, self.include_traceback
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_data
        )


def create_error_middleware():
    """Create error handling middleware"""
    
    async def error_middleware(request: Request, call_next):
        """Error handling middleware"""
        # Add request ID for tracking
        request.state.request_id = str(uuid.uuid4())
        
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # This catches any unhandled exceptions
            logger.error(
                f"Middleware caught unhandled exception: {type(exc).__name__}: {exc}",
                extra={
                    "request_id": request.state.request_id,
                    "request_url": str(request.url)
                },
                exc_info=True
            )
            
            # Convert to EmbeddingServerError for consistent handling
            server_error = EmbeddingServerError(
                message="Internal server error occurred",
                error_code="MIDDLEWARE_ERROR",
                details={"original_error": str(exc)}
            )
            
            error_data, status_code = ErrorResponse.create_response(
                server_error, request, include_traceback=False
            )
            
            return JSONResponse(
                status_code=status_code,
                content=error_data
            )
    
    return error_middleware


def setup_error_handlers(app, include_traceback: bool = False):
    """Setup error handlers for FastAPI app"""
    handlers = ErrorHandlers(include_traceback=include_traceback)
    
    # Register exception handlers
    app.add_exception_handler(EmbeddingServerError, handlers.embedding_server_error_handler)
    app.add_exception_handler(HTTPException, handlers.http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, handlers.http_exception_handler)
    app.add_exception_handler(RequestValidationError, handlers.validation_error_handler)
    app.add_exception_handler(Exception, handlers.generic_exception_handler)
    
    # Add error middleware
    app.middleware("http")(create_error_middleware())
    
    logger.info("Error handlers configured")


# Utility functions for raising common errors
def raise_model_not_ready():
    """Raise model not ready error"""
    raise ModelNotReadyError()


def raise_validation_error(message: str, field: str = None, value: Any = None):
    """Raise validation error"""
    raise ValidationError(message, field, value)


def raise_inference_error(message: str, original_error: Exception = None):
    """Raise model inference error"""
    raise ModelInferenceError(message, original_error)


def raise_resource_exhausted(message: str, resource_type: str = "memory"):
    """Raise resource exhausted error"""
    raise ResourceExhaustedError(message, resource_type)


def raise_configuration_error(message: str, config_key: str = None):
    """Raise configuration error"""
    raise ConfigurationError(message, config_key)


# Context manager for safe model operations
class SafeModelOperation:
    """Context manager for safe model operations with error handling"""
    
    def __init__(self, operation_name: str, model_instance=None):
        self.operation_name = operation_name
        self.model_instance = model_instance
    
    def __enter__(self):
        if self.model_instance and not self.model_instance.is_ready():
            raise_model_not_ready()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error in {self.operation_name}: {exc_val}")
            if isinstance(exc_val, (RuntimeError, MemoryError, OSError)):
                raise_inference_error(
                    f"Model operation failed: {self.operation_name}",
                    exc_val
                )
        return False  # Don't suppress exceptions
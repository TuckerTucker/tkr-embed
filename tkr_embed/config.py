"""
Configuration management for MLX text generation server
Supports environment variables, config files, and runtime configuration
"""

import os
import json
import logging
import psutil
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "INFO"
    access_log: bool = True
    timeout_keep_alive: int = 65


@dataclass
class ModelConfig:
    """Model configuration for text generation"""
    model_path: str = "openai/gpt-oss-20b"
    quantization: str = "auto"  # auto, q4, q8, none
    cache_dir: str = "./models"
    device: str = "auto"  # auto, cpu, gpu
    max_sequence_length: int = 8192
    trust_remote_code: bool = True

    # Text generation specific settings
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Reasoning configuration
    reasoning_levels: Dict[str, int] = field(default_factory=lambda: {
        "basic": 1,
        "detailed": 3,
        "deep": 5,
        "expert": 7
    })
    default_reasoning_level: str = "detailed"

    # Memory management for 21B model
    gpu_memory_fraction: float = 0.85
    cpu_memory_limit_gb: Optional[int] = None  # Auto-detect based on system

    def __post_init__(self):
        """Auto-configure memory settings based on system resources"""
        if self.cpu_memory_limit_gb is None:
            # Auto-detect and set conservative memory limit
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            self.cpu_memory_limit_gb = int(total_memory_gb * 0.7)  # Use 70% of available RAM


@dataclass
class SecurityConfig:
    """Security configuration"""
    require_api_key: bool = True
    require_https: bool = False
    api_key_header: str = "X-API-Key"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10


@dataclass
class GenerationConfig:
    """Text generation specific configuration"""
    streaming_enabled: bool = True
    streaming_chunk_size: int = 256
    generation_timeout_seconds: int = 300
    max_concurrent_generations: int = 10
    enable_chat_mode: bool = True
    enable_completion_mode: bool = True

    # Token limits and safety
    max_total_tokens: int = 8192  # Combined prompt + completion
    safety_filter_enabled: bool = True
    content_filter_threshold: float = 0.8


@dataclass
class QuantizationConfig:
    """Advanced quantization configuration for 21B model"""
    auto_quantization_enabled: bool = True
    memory_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "q4": 16,    # Use Q4 if less than 16GB RAM
        "q8": 32,    # Use Q8 if less than 32GB RAM
        "none": 64   # No quantization if 64GB+ RAM
    })

    # Model-specific quantization settings
    q4_group_size: int = 128
    q8_group_size: int = 256
    calibration_dataset_size: int = 512

    def get_recommended_quantization(self) -> str:
        """Get recommended quantization based on system memory"""
        if not self.auto_quantization_enabled:
            return "none"

        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if available_memory_gb < self.memory_thresholds["q4"]:
            return "q4"
        elif available_memory_gb < self.memory_thresholds["q8"]:
            return "q8"
        else:
            return "none"


@dataclass
class CacheConfig:
    """Caching configuration optimized for generation workloads"""
    enabled: bool = True
    max_size: int = 500  # Reduced for generation responses
    ttl_seconds: int = 1800  # Shorter TTL for dynamic content
    cleanup_interval_seconds: int = 300

    # Generation-specific caching
    cache_completions: bool = True
    cache_reasoning_steps: bool = False  # Don't cache reasoning for freshness
    max_response_cache_size_kb: int = 100  # Max size of cached responses


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    json_format: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    performance_tracking: bool = True


@dataclass
class Config:
    """Main configuration class"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Component configurations
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization configuration adjustments"""
        # Adjust settings based on environment
        if self.environment == Environment.PRODUCTION:
            self.debug = False
            self.server.reload = False
            self.security.require_https = True
            self.logging.level = "WARNING"
            self.generation.safety_filter_enabled = True
            self.generation.max_concurrent_generations = 20
        elif self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.server.reload = True
            self.security.require_https = False
            self.logging.level = "DEBUG"
            self.generation.safety_filter_enabled = False
            self.generation.max_concurrent_generations = 5
        elif self.environment == Environment.TESTING:
            self.debug = True
            self.server.port = 8001  # Different port for testing
            self.cache.enabled = False  # Disable cache for consistent tests
            self.rate_limit.enabled = False  # Disable rate limiting for tests
            self.generation.safety_filter_enabled = False
            self.generation.max_concurrent_generations = 2
            self.generation.generation_timeout_seconds = 60


class ConfigManager:
    """Configuration manager with multiple sources"""
    
    def __init__(self):
        self.config = Config()
        self._config_sources = []
    
    def load_from_environment(self) -> 'ConfigManager':
        """Load configuration from environment variables"""
        
        # Environment
        if env := os.getenv("GENERATION_ENV"):
            try:
                self.config.environment = Environment(env.lower())
            except ValueError:
                logger.warning(f"Invalid environment: {env}")

        # Debug mode
        if debug := os.getenv("GENERATION_DEBUG"):
            self.config.debug = debug.lower() in ("true", "1", "yes")
        
        # Server configuration
        if host := os.getenv("GENERATION_HOST"):
            self.config.server.host = host
        if port := os.getenv("GENERATION_PORT"):
            self.config.server.port = int(port)
        if workers := os.getenv("GENERATION_WORKERS"):
            self.config.server.workers = int(workers)
        if log_level := os.getenv("GENERATION_LOG_LEVEL"):
            self.config.server.log_level = log_level.upper()
        
        # Model configuration
        if model_path := os.getenv("GENERATION_MODEL_PATH"):
            self.config.model.model_path = model_path
        if quantization := os.getenv("GENERATION_QUANTIZATION"):
            self.config.model.quantization = quantization
        if cache_dir := os.getenv("GENERATION_CACHE_DIR"):
            self.config.model.cache_dir = cache_dir
        if device := os.getenv("GENERATION_DEVICE"):
            self.config.model.device = device
        if max_tokens := os.getenv("GENERATION_MAX_TOKENS"):
            self.config.model.max_tokens = int(max_tokens)
        if temperature := os.getenv("GENERATION_TEMPERATURE"):
            self.config.model.temperature = float(temperature)

        # Generation configuration
        if streaming := os.getenv("GENERATION_STREAMING_ENABLED"):
            self.config.generation.streaming_enabled = streaming.lower() in ("true", "1", "yes")
        if timeout := os.getenv("GENERATION_TIMEOUT"):
            self.config.generation.generation_timeout_seconds = int(timeout)
        if max_concurrent := os.getenv("GENERATION_MAX_CONCURRENT"):
            self.config.generation.max_concurrent_generations = int(max_concurrent)
        
        # Security configuration
        if require_https := os.getenv("GENERATION_REQUIRE_HTTPS"):
            self.config.security.require_https = require_https.lower() in ("true", "1", "yes")
        if cors_origins := os.getenv("GENERATION_CORS_ORIGINS"):
            self.config.security.cors_origins = cors_origins.split(",")
        
        # Rate limiting
        if rate_limit_enabled := os.getenv("GENERATION_RATE_LIMIT_ENABLED"):
            self.config.rate_limit.enabled = rate_limit_enabled.lower() in ("true", "1", "yes")
        if rpm := os.getenv("GENERATION_RATE_LIMIT_RPM"):
            self.config.rate_limit.requests_per_minute = int(rpm)
        if rph := os.getenv("GENERATION_RATE_LIMIT_RPH"):
            self.config.rate_limit.requests_per_hour = int(rph)
        
        # Cache configuration
        if cache_enabled := os.getenv("GENERATION_CACHE_ENABLED"):
            self.config.cache.enabled = cache_enabled.lower() in ("true", "1", "yes")
        if cache_size := os.getenv("GENERATION_CACHE_SIZE"):
            self.config.cache.max_size = int(cache_size)
        if cache_ttl := os.getenv("GENERATION_CACHE_TTL"):
            self.config.cache.ttl_seconds = int(cache_ttl)
        
        # Logging configuration
        if log_file := os.getenv("GENERATION_LOG_FILE"):
            self.config.logging.file_path = log_file
        if log_format := os.getenv("GENERATION_LOG_FORMAT"):
            self.config.logging.json_format = log_format.lower() == "json"
        
        self._config_sources.append("environment")
        logger.info("Configuration loaded from environment variables")
        return self
    
    def load_from_file(self, file_path: Union[str, Path]) -> 'ConfigManager':
        """Load configuration from YAML or JSON file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return self
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {file_path.suffix}")
                    return self
            
            self._apply_config_data(data)
            self._config_sources.append(f"file:{file_path}")
            logger.info(f"Configuration loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
        
        return self
    
    def load_from_dict(self, data: Dict[str, Any]) -> 'ConfigManager':
        """Load configuration from dictionary"""
        self._apply_config_data(data)
        self._config_sources.append("dictionary")
        logger.info("Configuration loaded from dictionary")
        return self
    
    def _apply_config_data(self, data: Dict[str, Any]):
        """Apply configuration data to config object"""
        
        if "environment" in data:
            try:
                self.config.environment = Environment(data["environment"].lower())
            except ValueError:
                logger.warning(f"Invalid environment in config: {data['environment']}")
        
        if "debug" in data:
            self.config.debug = bool(data["debug"])
        
        # Update server config
        if "server" in data:
            server_data = data["server"]
            for key, value in server_data.items():
                if hasattr(self.config.server, key):
                    setattr(self.config.server, key, value)
        
        # Update model config
        if "model" in data:
            model_data = data["model"]
            for key, value in model_data.items():
                if hasattr(self.config.model, key):
                    setattr(self.config.model, key, value)

        # Update generation config
        if "generation" in data:
            generation_data = data["generation"]
            for key, value in generation_data.items():
                if hasattr(self.config.generation, key):
                    setattr(self.config.generation, key, value)

        # Update quantization config
        if "quantization" in data:
            quantization_data = data["quantization"]
            for key, value in quantization_data.items():
                if hasattr(self.config.quantization, key):
                    setattr(self.config.quantization, key, value)
        
        # Update security config
        if "security" in data:
            security_data = data["security"]
            for key, value in security_data.items():
                if hasattr(self.config.security, key):
                    setattr(self.config.security, key, value)
        
        # Update rate limit config
        if "rate_limit" in data:
            rate_limit_data = data["rate_limit"]
            for key, value in rate_limit_data.items():
                if hasattr(self.config.rate_limit, key):
                    setattr(self.config.rate_limit, key, value)
        
        # Update cache config
        if "cache" in data:
            cache_data = data["cache"]
            for key, value in cache_data.items():
                if hasattr(self.config.cache, key):
                    setattr(self.config.cache, key, value)
        
        # Update logging config
        if "logging" in data:
            logging_data = data["logging"]
            for key, value in logging_data.items():
                if hasattr(self.config.logging, key):
                    setattr(self.config.logging, key, value)
        
        # Update monitoring config
        if "monitoring" in data:
            monitoring_data = data["monitoring"]
            for key, value in monitoring_data.items():
                if hasattr(self.config.monitoring, key):
                    setattr(self.config.monitoring, key, value)
        
        # Store custom settings
        if "custom" in data:
            self.config.custom.update(data["custom"])
    
    def get_config(self) -> Config:
        """Get the current configuration"""
        return self.config
    
    def get_config_sources(self) -> List[str]:
        """Get list of configuration sources"""
        return self._config_sources.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.config.environment.value,
            "debug": self.config.debug,
            "server": {
                "host": self.config.server.host,
                "port": self.config.server.port,
                "workers": self.config.server.workers,
                "reload": self.config.server.reload,
                "log_level": self.config.server.log_level,
                "access_log": self.config.server.access_log,
                "timeout_keep_alive": self.config.server.timeout_keep_alive
            },
            "model": {
                "model_path": self.config.model.model_path,
                "quantization": self.config.model.quantization,
                "cache_dir": self.config.model.cache_dir,
                "device": self.config.model.device,
                "max_sequence_length": self.config.model.max_sequence_length,
                "trust_remote_code": self.config.model.trust_remote_code,
                "max_tokens": self.config.model.max_tokens,
                "temperature": self.config.model.temperature,
                "top_p": self.config.model.top_p,
                "top_k": self.config.model.top_k,
                "repetition_penalty": self.config.model.repetition_penalty,
                "reasoning_levels": self.config.model.reasoning_levels,
                "default_reasoning_level": self.config.model.default_reasoning_level,
                "gpu_memory_fraction": self.config.model.gpu_memory_fraction,
                "cpu_memory_limit_gb": self.config.model.cpu_memory_limit_gb
            },
            "generation": {
                "streaming_enabled": self.config.generation.streaming_enabled,
                "streaming_chunk_size": self.config.generation.streaming_chunk_size,
                "generation_timeout_seconds": self.config.generation.generation_timeout_seconds,
                "max_concurrent_generations": self.config.generation.max_concurrent_generations,
                "enable_chat_mode": self.config.generation.enable_chat_mode,
                "enable_completion_mode": self.config.generation.enable_completion_mode,
                "max_total_tokens": self.config.generation.max_total_tokens,
                "safety_filter_enabled": self.config.generation.safety_filter_enabled,
                "content_filter_threshold": self.config.generation.content_filter_threshold
            },
            "quantization": {
                "auto_quantization_enabled": self.config.quantization.auto_quantization_enabled,
                "memory_thresholds": self.config.quantization.memory_thresholds,
                "q4_group_size": self.config.quantization.q4_group_size,
                "q8_group_size": self.config.quantization.q8_group_size,
                "calibration_dataset_size": self.config.quantization.calibration_dataset_size
            },
            "security": {
                "require_api_key": self.config.security.require_api_key,
                "require_https": self.config.security.require_https,
                "api_key_header": self.config.security.api_key_header,
                "cors_origins": self.config.security.cors_origins,
                "cors_methods": self.config.security.cors_methods,
                "cors_headers": self.config.security.cors_headers
            },
            "rate_limit": {
                "enabled": self.config.rate_limit.enabled,
                "requests_per_minute": self.config.rate_limit.requests_per_minute,
                "requests_per_hour": self.config.rate_limit.requests_per_hour,
                "requests_per_day": self.config.rate_limit.requests_per_day,
                "burst_size": self.config.rate_limit.burst_size
            },
            "cache": {
                "enabled": self.config.cache.enabled,
                "max_size": self.config.cache.max_size,
                "ttl_seconds": self.config.cache.ttl_seconds,
                "cleanup_interval_seconds": self.config.cache.cleanup_interval_seconds,
                "cache_completions": self.config.cache.cache_completions,
                "cache_reasoning_steps": self.config.cache.cache_reasoning_steps,
                "max_response_cache_size_kb": self.config.cache.max_response_cache_size_kb
            },
            "logging": {
                "level": self.config.logging.level,
                "format": self.config.logging.format,
                "file_path": self.config.logging.file_path,
                "max_file_size": self.config.logging.max_file_size,
                "backup_count": self.config.logging.backup_count,
                "json_format": self.config.logging.json_format
            },
            "monitoring": {
                "enabled": self.config.monitoring.enabled,
                "metrics_port": self.config.monitoring.metrics_port,
                "health_check_interval": self.config.monitoring.health_check_interval,
                "performance_tracking": self.config.monitoring.performance_tracking
            },
            "custom": self.config.custom,
            "config_sources": self._config_sources
        }
    
    def save_to_file(self, file_path: Union[str, Path], format_type: str = "yaml"):
        """Save current configuration to file"""
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        try:
            with open(file_path, 'w') as f:
                if format_type.lower() == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif format_type.lower() == "json":
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")


# Global configuration manager
config_manager = ConfigManager()

# Load configuration from environment by default
config_manager.load_from_environment()

# Try to load from config file
# Priority order: development configs first, then production configs
config_file_paths = [
    "config.dev.yaml",      # Development-specific config
    "config.dev.yml",
    "config.yaml",          # Production config
    "config.yml",
    "config.json",
    ".config/generation-server.yaml",
    ".config/generation-server.yml",
    os.path.expanduser("~/.config/generation-server.yaml")
]

for config_path in config_file_paths:
    if os.path.exists(config_path):
        config_manager.load_from_file(config_path)
        logger.info(f"Using configuration file: {config_path}")
        break

# Global config instance
config = config_manager.get_config()


def get_config() -> Config:
    """Get global configuration"""
    return config


def reload_config():
    """Reload configuration from all sources"""
    global config, config_manager
    config_manager = ConfigManager()
    config_manager.load_from_environment()
    
    for config_path in config_file_paths:
        if os.path.exists(config_path):
            config_manager.load_from_file(config_path)
            logger.info(f"Configuration reloaded from: {config_path}")
            break
    
    config = config_manager.get_config()
    logger.info("Configuration reloaded")


if __name__ == "__main__":
    # Print current configuration
    import pprint
    print("Current Configuration:")
    pprint.pprint(config_manager.to_dict())
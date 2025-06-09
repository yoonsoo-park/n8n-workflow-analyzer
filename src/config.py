"""
Configuration management for the n8n workflow analyzer.

This module handles configuration settings, logging setup,
and environment variables for the application.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class AnalysisConfig:
    """Configuration for workflow analysis parameters."""
    
    # Pattern mining settings
    min_support: float = 0.3  # Increased from 0.1 for smaller datasets
    min_confidence: float = 0.6  # Increased from 0.5
    max_itemsets: int = 100  # Reduced from 1000
    
    # Network analysis settings
    centrality_algorithms: list = None
    community_detection_algorithm: str = "louvain"
    max_path_length: int = 10
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 100
    
    # Visualization settings
    default_layout: str = "spring"
    node_size_range: tuple = (10, 100)
    edge_width_range: tuple = (1, 5)
    
    def __post_init__(self):
        if self.centrality_algorithms is None:
            self.centrality_algorithms = ["degree", "betweenness", "closeness", "eigenvector"]


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "n8n_analysis"
    username: str = "analyzer"
    password: str = ""
    connection_timeout: int = 30


@dataclass
class WebConfig:
    """Configuration for web application settings."""
    
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    secret_key: str = "dev-secret-key"
    upload_folder: str = "uploads"
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: set = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {"json", "zip"}


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class ConfigManager:
    """Manages application configuration from multiple sources."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        self.analysis_config = AnalysisConfig()
        self.database_config = DatabaseConfig()
        self.web_config = WebConfig()
        self.logging_config = LoggingConfig()
        
        self._load_configuration()
        self._setup_logging()
    
    def _load_configuration(self):
        """Load configuration from file and environment variables."""
        # Load from file if provided
        if self.config_file and Path(self.config_file).exists():
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_environment()
    
    def _load_from_file(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update analysis config
            if 'analysis' in config_data:
                self._update_dataclass(self.analysis_config, config_data['analysis'])
            
            # Update database config
            if 'database' in config_data:
                self._update_dataclass(self.database_config, config_data['database'])
            
            # Update web config
            if 'web' in config_data:
                self._update_dataclass(self.web_config, config_data['web'])
            
            # Update logging config
            if 'logging' in config_data:
                self._update_dataclass(self.logging_config, config_data['logging'])
                
        except Exception as e:
            print(f"Warning: Could not load configuration file {self.config_file}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Database configuration
        if os.getenv('DB_HOST'):
            self.database_config.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            self.database_config.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            self.database_config.database = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            self.database_config.username = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            self.database_config.password = os.getenv('DB_PASSWORD')
        
        # Web configuration
        if os.getenv('WEB_HOST'):
            self.web_config.host = os.getenv('WEB_HOST')
        if os.getenv('WEB_PORT'):
            self.web_config.port = int(os.getenv('WEB_PORT'))
        if os.getenv('WEB_DEBUG'):
            self.web_config.debug = os.getenv('WEB_DEBUG').lower() == 'true'
        if os.getenv('SECRET_KEY'):
            self.web_config.secret_key = os.getenv('SECRET_KEY')
        
        # Logging configuration
        if os.getenv('LOG_LEVEL'):
            self.logging_config.level = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.logging_config.file_path = os.getenv('LOG_FILE')
    
    def _update_dataclass(self, dataclass_instance, config_dict):
        """Update a dataclass instance with values from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        # Convert string level to logging constant
        level = getattr(logging, self.logging_config.level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(self.logging_config.format)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if specified
        if self.logging_config.file_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging_config.file_path,
                maxBytes=self.logging_config.max_file_size,
                backupCount=self.logging_config.backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def save_configuration(self, output_file: str):
        """Save current configuration to a YAML file."""
        config_data = {
            'analysis': asdict(self.analysis_config),
            'database': asdict(self.database_config),
            'web': asdict(self.web_config),
            'logging': asdict(self.logging_config)
        }
        
        # Convert sets to lists for YAML serialization
        if 'allowed_extensions' in config_data['web']:
            config_data['web']['allowed_extensions'] = list(config_data['web']['allowed_extensions'])
        
        with open(output_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_analysis_config(self) -> AnalysisConfig:
        """Get the analysis configuration."""
        return self.analysis_config
    
    def get_database_config(self) -> DatabaseConfig:
        """Get the database configuration."""
        return self.database_config
    
    def get_web_config(self) -> WebConfig:
        """Get the web configuration."""
        return self.web_config
    
    def get_logging_config(self) -> LoggingConfig:
        """Get the logging configuration."""
        return self.logging_config
    
    def update_analysis_config(self, **kwargs):
        """Update analysis configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.analysis_config, key):
                setattr(self.analysis_config, key, value)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of all configuration settings."""
        return {
            'analysis': asdict(self.analysis_config),
            'database': asdict(self.database_config),
            'web': asdict(self.web_config),
            'logging': asdict(self.logging_config)
        }


# Global configuration instance
_config_manager = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_analysis_config() -> AnalysisConfig:
    """Get the analysis configuration."""
    return get_config_manager().get_analysis_config()


def get_database_config() -> DatabaseConfig:
    """Get the database configuration."""
    return get_config_manager().get_database_config()


def get_web_config() -> WebConfig:
    """Get the web configuration."""
    return get_config_manager().get_web_config()


def get_logging_config() -> LoggingConfig:
    """Get the logging configuration."""
    return get_config_manager().get_logging_config()


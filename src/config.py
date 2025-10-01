"""
Configuration Management Module
Handles all application configuration and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AppConfig:
    """Application configuration"""
    name: str = "GenAI Autonomous Data Agent"
    version: str = "1.0.0"
    description: str = "AI-powered data analysis"
    environment: str = "development"

@dataclass
class GPUConfig:
    """GPU and acceleration configuration"""
    enabled: bool = False
    device: str = "cuda:0"
    memory_fraction: float = 0.8

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"
    sqlite_path: str = "./data/autonomous_agent.db"

@dataclass
class MLConfig:
    """Machine learning configuration"""
    algorithms: list = None
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression"]

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._load_config()
        self._setup_paths()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as file:
                    config_data = yaml.safe_load(file)
            else:
                config_data = self._get_default_config()
            
            # Initialize configuration objects
            self.app = AppConfig(**config_data.get('app', {}))
            self.gpu = GPUConfig(**config_data.get('gpu', {}))
            self.database = DatabaseConfig(**config_data.get('database', {}))
            self.ml = MLConfig(**config_data.get('ml', {}))
            
            # Store raw config for additional settings
            self.raw_config = config_data
            
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            self._load_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'app': {'name': 'GenAI Autonomous Data Agent', 'version': '1.0.0'},
            'gpu': {'enabled': False},
            'database': {'type': 'sqlite', 'sqlite_path': './data/autonomous_agent.db'},
            'ml': {'test_size': 0.2, 'cv_folds': 5, 'random_state': 42}
        }
    
    def _load_default_config(self):
        """Load default configuration as fallback"""
        default_config = self._get_default_config()
        self.app = AppConfig(**default_config['app'])
        self.gpu = GPUConfig(**default_config['gpu'])
        self.database = DatabaseConfig(**default_config['database'])
        self.ml = MLConfig(**default_config['ml'])
        self.raw_config = default_config
    
    def _setup_paths(self):
        """Setup and create necessary directories"""
        directories = [
            "data/raw",
            "data/processed", 
            "data/sample_datasets",
            "models/trained_models",
            "models/explainers",
            "models/artifacts",
            "outputs/reports",
            "outputs/visualizations",
            "outputs/exports",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key from environment variables"""
        key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY', 
            'nvidia': 'NVIDIA_API_KEY'
        }
        
        env_var = key_mapping.get(service.lower())
        if env_var:
            return os.getenv(env_var)
        return None
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available and enabled"""
        if not self.gpu.enabled:
            return False
            
        try:
            import cudf
            return True
        except ImportError:
            return False

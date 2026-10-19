"""
Configuration management module
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """RM Abstract Layer configuration"""

    # Device settings
    device: str = "auto"

    # Cache settings
    cache_dir: Optional[str] = None

    # NPU compilation options
    compile_options: Dict[str, Any] = field(default_factory=dict)

    # Logging settings
    verbose: bool = True

    # Inference engine options (vLLM, etc.)
    engine_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set default cache directory
        if self.cache_dir is None:
            self.cache_dir = os.path.join(Path.home(), ".rm_abstract", "cache")

        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def device_type(self) -> str:
        """Parse device type (gpu, rbln, furiosa, cpu, auto)"""
        if self.device == "auto":
            return "auto"
        return self.device.split(":")[0].lower()

    @property
    def device_id(self) -> int:
        """Parse device ID"""
        if self.device == "auto" or ":" not in self.device:
            return 0
        try:
            return int(self.device.split(":")[1])
        except (ValueError, IndexError):
            return 0

    def get_compile_option(self, key: str, default: Any = None) -> Any:
        """Get compile option"""
        return self.compile_options.get(key, default)

    def get_engine_option(self, key: str, default: Any = None) -> Any:
        """Get engine option"""
        return self.engine_options.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "device": self.device,
            "cache_dir": self.cache_dir,
            "compile_options": self.compile_options,
            "verbose": self.verbose,
            "engine_options": self.engine_options,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            device=os.environ.get("RM_DEVICE", "auto"),
            cache_dir=os.environ.get("RM_CACHE_DIR"),
            verbose=os.environ.get("RM_VERBOSE", "true").lower() == "true",
        )

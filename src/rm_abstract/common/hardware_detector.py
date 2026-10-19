"""
Hardware Detector

Unified hardware detection utilities.
Detects available devices (GPU, NPU, CPU) across different vendors.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Hardware type enumeration"""

    GPU = "gpu"
    NPU = "npu"
    CPU = "cpu"


class HardwareVendor(Enum):
    """Hardware vendor enumeration"""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    REBELLIONS = "rebellions"
    FURIOSA = "furiosa"
    GENERIC = "generic"


@dataclass
class HardwareInfo:
    """Hardware information container"""

    hardware_type: HardwareType
    vendor: HardwareVendor
    device_id: int
    name: str
    available: bool = True
    memory_total: Optional[int] = None  # bytes
    memory_available: Optional[int] = None  # bytes
    sdk_version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def device_string(self) -> str:
        """Return device string for backend selection"""
        vendor_map = {
            HardwareVendor.NVIDIA: "gpu",
            HardwareVendor.AMD: "gpu",
            HardwareVendor.REBELLIONS: "rbln",
            HardwareVendor.FURIOSA: "furiosa",
            HardwareVendor.GENERIC: "cpu",
        }
        return vendor_map.get(self.vendor, "cpu")


@dataclass
class DetectorConfig:
    """Configuration for hardware detector"""

    # SDK import names (for dynamic detection)
    # Format: {"vendor_name": ["package1", "package2"]}
    sdk_packages: Dict[str, List[str]] = field(default_factory=dict)

    # Detection functions (for custom detection)
    # Format: {"vendor_name": callable}
    custom_detectors: Dict[str, Callable] = field(default_factory=dict)


class HardwareDetector:
    """
    Unified hardware detector.

    Detects available hardware across different vendors without
    hardcoding SDK import names throughout the codebase.

    Example:
        detector = HardwareDetector()

        # Get all available hardware
        hardware_list = detector.detect_all()

        # Check specific hardware
        if detector.is_available("nvidia"):
            print("NVIDIA GPU available")

        # Get best available device
        best = detector.get_best_available()
    """

    # Default SDK package names for each vendor
    DEFAULT_SDK_PACKAGES = {
        "nvidia": ["torch"],
        "amd": ["torch"],  # ROCm uses same torch API
        "rebellions": ["rebel", "rbln", "vllm_rbln", "optimum.rbln"],
        "furiosa": ["furiosa", "furiosa.runtime"],
    }

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Args:
            config: Optional detector configuration
        """
        self.config = config or DetectorConfig()
        self._cache: Dict[str, List[HardwareInfo]] = {}

        # Merge default SDK packages with config
        self.sdk_packages = {**self.DEFAULT_SDK_PACKAGES}
        if self.config.sdk_packages:
            self.sdk_packages.update(self.config.sdk_packages)

    def detect_all(self, use_cache: bool = True) -> List[HardwareInfo]:
        """
        Detect all available hardware.

        Args:
            use_cache: Whether to use cached results

        Returns:
            List of HardwareInfo for all detected hardware
        """
        if use_cache and "all" in self._cache:
            return self._cache["all"]

        hardware_list = []

        # Detect GPUs
        hardware_list.extend(self._detect_nvidia_gpu())
        hardware_list.extend(self._detect_amd_gpu())

        # Detect NPUs
        hardware_list.extend(self._detect_rebellions_npu())
        hardware_list.extend(self._detect_furiosa_npu())

        # CPU is always available
        hardware_list.extend(self._detect_cpu())

        self._cache["all"] = hardware_list
        return hardware_list

    def is_available(self, vendor: str) -> bool:
        """
        Check if hardware from a specific vendor is available.

        Args:
            vendor: Vendor name (nvidia, amd, rebellions, furiosa, cpu)

        Returns:
            True if available
        """
        vendor = vendor.lower()

        if vendor in ("nvidia", "gpu"):
            return len(self._detect_nvidia_gpu()) > 0
        elif vendor == "amd":
            return len(self._detect_amd_gpu()) > 0
        elif vendor in ("rebellions", "rbln"):
            return len(self._detect_rebellions_npu()) > 0
        elif vendor == "furiosa":
            return len(self._detect_furiosa_npu()) > 0
        elif vendor == "cpu":
            return True

        return False

    def get_best_available(
        self,
        priority: Optional[List[str]] = None,
    ) -> Optional[HardwareInfo]:
        """
        Get the best available hardware based on priority.

        Args:
            priority: Priority order (default: NPU > GPU > CPU)

        Returns:
            Best available HardwareInfo or None
        """
        if priority is None:
            priority = ["rebellions", "furiosa", "nvidia", "amd", "cpu"]

        for vendor in priority:
            if self.is_available(vendor):
                hardware_list = self.get_by_vendor(vendor)
                if hardware_list:
                    return hardware_list[0]

        return None

    def get_by_vendor(self, vendor: str) -> List[HardwareInfo]:
        """
        Get hardware info for a specific vendor.

        Args:
            vendor: Vendor name

        Returns:
            List of HardwareInfo
        """
        vendor = vendor.lower()

        if vendor in ("nvidia", "gpu"):
            return self._detect_nvidia_gpu()
        elif vendor == "amd":
            return self._detect_amd_gpu()
        elif vendor in ("rebellions", "rbln"):
            return self._detect_rebellions_npu()
        elif vendor == "furiosa":
            return self._detect_furiosa_npu()
        elif vendor == "cpu":
            return self._detect_cpu()

        return []

    def get_by_type(self, hardware_type: HardwareType) -> List[HardwareInfo]:
        """
        Get hardware info by type.

        Args:
            hardware_type: HardwareType enum

        Returns:
            List of HardwareInfo
        """
        all_hardware = self.detect_all()
        return [h for h in all_hardware if h.hardware_type == hardware_type]

    def _detect_nvidia_gpu(self) -> List[HardwareInfo]:
        """Detect NVIDIA GPUs"""
        hardware_list = []

        try:
            import torch

            if not torch.cuda.is_available():
                return []

            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                hardware_list.append(
                    HardwareInfo(
                        hardware_type=HardwareType.GPU,
                        vendor=HardwareVendor.NVIDIA,
                        device_id=i,
                        name=props.name,
                        memory_total=props.total_memory,
                        extra={
                            "compute_capability": f"{props.major}.{props.minor}",
                            "multi_processor_count": props.multi_processor_count,
                        },
                    )
                )

        except ImportError:
            logger.debug("PyTorch not installed, skipping NVIDIA GPU detection")
        except Exception as e:
            logger.debug(f"NVIDIA GPU detection failed: {e}")

        return hardware_list

    def _detect_amd_gpu(self) -> List[HardwareInfo]:
        """Detect AMD GPUs (ROCm)"""
        hardware_list = []

        try:
            import torch

            # Check for ROCm
            if not (hasattr(torch, "hip") or "rocm" in torch.__version__.lower()):
                return []

            if not torch.cuda.is_available():  # ROCm uses CUDA API
                return []

            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                hardware_list.append(
                    HardwareInfo(
                        hardware_type=HardwareType.GPU,
                        vendor=HardwareVendor.AMD,
                        device_id=i,
                        name=props.name,
                        memory_total=props.total_memory,
                    )
                )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"AMD GPU detection failed: {e}")

        return hardware_list

    def _detect_rebellions_npu(self) -> List[HardwareInfo]:
        """Detect Rebellions ATOM NPUs"""
        hardware_list = []

        # Try rebel SDK first
        try:
            import rebel

            devices = rebel.get_devices()
            sdk_version = getattr(rebel, "__version__", "unknown")

            for i, device in enumerate(devices):
                hardware_list.append(
                    HardwareInfo(
                        hardware_type=HardwareType.NPU,
                        vendor=HardwareVendor.REBELLIONS,
                        device_id=i,
                        name="Rebellions ATOM",
                        sdk_version=sdk_version,
                        extra={"device_info": str(device)},
                    )
                )
            return hardware_list

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Rebellions (rebel) detection failed: {e}")

        # Try rbln SDK
        try:
            import rbln

            devices = rbln.get_devices()
            sdk_version = getattr(rbln, "__version__", "unknown")

            for i, device in enumerate(devices):
                hardware_list.append(
                    HardwareInfo(
                        hardware_type=HardwareType.NPU,
                        vendor=HardwareVendor.REBELLIONS,
                        device_id=i,
                        name="Rebellions ATOM",
                        sdk_version=sdk_version,
                        extra={"device_info": str(device)},
                    )
                )
            return hardware_list

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Rebellions (rbln) detection failed: {e}")

        return hardware_list

    def _detect_furiosa_npu(self) -> List[HardwareInfo]:
        """Detect FuriosaAI RNGD NPUs"""
        hardware_list = []

        try:
            from furiosa import runtime

            devices = runtime.list_devices()

            for i, device in enumerate(devices):
                hardware_list.append(
                    HardwareInfo(
                        hardware_type=HardwareType.NPU,
                        vendor=HardwareVendor.FURIOSA,
                        device_id=i,
                        name="FuriosaAI RNGD",
                        extra={"device_info": str(device)},
                    )
                )

        except ImportError:
            logger.debug("Furiosa SDK not installed")
        except Exception as e:
            logger.debug(f"Furiosa NPU detection failed: {e}")

        return hardware_list

    def _detect_cpu(self) -> List[HardwareInfo]:
        """Detect CPU (always available)"""
        import os
        import platform

        return [
            HardwareInfo(
                hardware_type=HardwareType.CPU,
                vendor=HardwareVendor.GENERIC,
                device_id=0,
                name=platform.processor() or "CPU",
                extra={
                    "cpu_count": os.cpu_count(),
                    "platform": platform.platform(),
                },
            )
        ]

    def clear_cache(self) -> None:
        """Clear detection cache"""
        self._cache.clear()

    def summary(self) -> str:
        """
        Get a summary of detected hardware.

        Returns:
            Human-readable summary string
        """
        all_hardware = self.detect_all()

        if not all_hardware:
            return "No hardware detected"

        lines = ["Detected Hardware:"]
        for hw in all_hardware:
            status = "available" if hw.available else "unavailable"
            mem_str = ""
            if hw.memory_total:
                mem_gb = hw.memory_total / (1024**3)
                mem_str = f" ({mem_gb:.1f} GB)"

            lines.append(f"  - {hw.vendor.value} {hw.name}{mem_str} [{status}]")

        return "\n".join(lines)

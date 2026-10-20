"""
rm_core AcceleratorProvider for rm_abstract (ai-rm protocol v0.1, PROTOCOL.md section 3).

Device discovery per vendor, in order of preference:

1. The vendor's management tool (``nvidia-smi``, ``furiosa-smi``, ``rbln-stat``), read-only.
   It reports the device's own index, product name and memory without opening a runtime
   context on a device that may be busy.
2. The existing ``HardwareDetector`` (SDK/torch based) when the tool is not installed.

Each record is converted to an ``rm_core.Accelerator`` with id ``<node>/<vendor>:<index>``.
The node name comes from ``AIRM_NODE_NAME`` (set from ``spec.nodeName`` in the DaemonSet),
falling back to the hostname. CPUs are not reported: a node without accelerators yields ``[]``.

Per-card occupancy (``occupancy.py``) is added to each accelerator's ``extra``: ``inUse``,
``holders`` and, for a card held by a process outside Kubernetes, ``health: busy`` /
``healthy: "false"``. ``AIRM_OCCUPANCY=0`` turns it off.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field, replace
from typing import Any

from rm_core import Accelerator

from ..common.hardware_detector import HardwareDetector, HardwareInfo
from . import occupancy

logger = logging.getLogger(__name__)

_TOOL_TIMEOUT_S = 10

# Vendor-stated figures by (vendor, model). peak_tflops is dense peak per precision and is only
# listed where the vendor datasheet states it; anything not listed stays unknown (omitted).
# memory_bytes is used only when the device itself does not report its memory.
_SPECS: dict[tuple, dict[str, Any]] = {
    ("furiosa", "RNGD"): {
        "memory_bytes": 48 << 30,
        "memory_kind": "hbm",
        "peak_tflops": {"bf16": 256.0, "fp8": 512.0},
        "peak_source": "FuriosaAI RNGD datasheet (vendor-stated dense peak)",
    },
    ("rebellions", "ATOM"): {
        "memory_bytes": 16 << 30,
        "memory_kind": "gddr",
        "peak_tflops": {"fp16": 32.0, "int8": 128.0},
        "peak_source": "Rebellions ATOM product brief (vendor-stated dense peak)",
    },
}

# Rebellions board product names (as reported by rbln-stat) that carry one ATOM chip.
_REBELLIONS_MODELS = {"RBLN-CA22": "ATOM"}

# NVIDIA data-center parts with HBM; every other NVIDIA part is treated as GDDR.
_NVIDIA_HBM = re.compile(
    r"\b(?:[ABH]100|[AH]800|H200|GH200|GB200|B200|V100|P100|A30)\b"
)

_PCIE_GEN = {"8": "pcie3", "16": "pcie4", "32": "pcie5"}


@dataclass
class _Device:
    """What one discovery path found about one device, before conversion."""

    vendor: str
    index: int
    product: str
    memory_bytes: int | None = None
    power_limit_w: float | None = None
    pci_bdf: str = ""
    link_speed: str = ""
    source: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def node_name() -> str:
    """Node name for accelerator ids: AIRM_NODE_NAME, else hostname, as a DNS-1123 name."""
    raw = os.environ.get("AIRM_NODE_NAME") or socket.gethostname()
    name = re.sub(r"[^a-z0-9.-]+", "-", raw.strip().lower())
    name = re.sub(r"^[^a-z0-9]+", "", name)
    return name or "localhost"


def _run(cmd: list[str]) -> str | None:
    """stdout of a vendor tool, or None if it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_TOOL_TIMEOUT_S,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug(f"{cmd[0]} failed: {e}")
        return None
    return out.stdout.decode("utf-8", "replace")


def _number(text: str) -> float | None:
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else None


def _pcie(link_speed: str, pci_bdf: str) -> list[str]:
    """PCIe generation from the reported link speed or, failing that, sysfs."""
    if not link_speed and pci_bdf:
        try:
            with open(
                f"/sys/bus/pci/devices/{pci_bdf.lower()}/current_link_speed"
            ) as f:
                link_speed = f.read()
        except OSError:
            return []
    gts = _number(link_speed)
    gen = _PCIE_GEN.get(str(int(gts))) if gts else None
    return [gen] if gen else []


# --- vendor tools -------------------------------------------------------------------------


def _nvidia_smi() -> list[_Device]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,power.limit,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    for line in (out or "").splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 5 or not cols[0].isdigit():
            continue
        mem_mib = _number(cols[2])
        bdf = cols[4].lower()
        # nvidia-smi prints an 8-digit PCI domain; sysfs uses 4.
        if re.match(r"^0000[0-9a-f]{4}:", bdf):
            bdf = bdf[4:]
        devices.append(
            _Device(
                vendor="nvidia",
                index=int(cols[0]),
                product=cols[1],
                memory_bytes=int(mem_mib) << 20 if mem_mib else None,
                power_limit_w=_number(cols[3]),
                pci_bdf=bdf,
                source="nvidia-smi",
            )
        )
    return devices


def _furiosa_smi() -> list[_Device]:
    out = _run(["furiosa-smi", "info", "--format", "json"])
    if not out:
        return []
    try:
        entries = json.loads(out)
    except ValueError:
        return []
    devices = []
    for e in entries if isinstance(entries, list) else []:
        m = re.match(r"^npu(\d+)$", str(e.get("dev_name", "")))
        if not m:
            continue
        devices.append(
            _Device(
                vendor="furiosa",
                index=int(m.group(1)),
                product=str(e.get("arch") or "RNGD"),
                power_limit_w=_number(str(e.get("power_limit", ""))),
                pci_bdf=str(e.get("pci_bdf", "")),
                source="furiosa-smi",
                extra={"firmware": str(e.get("firmware", ""))},
            )
        )
    return devices


def _rbln_stat() -> list[_Device]:
    out = _run(["rbln-stat", "--json"])
    if not out:
        return []
    try:
        entries = json.loads(out).get("devices", [])
    except (ValueError, AttributeError):
        return []
    devices = []
    for e in entries:
        # "npu" restarts at 0 in each device group; the device node (rbln<N>) is unique per host.
        m = re.fullmatch(r"rbln(\d+)", str(e.get("device", "")))
        index = m.group(1) if m else str(e.get("npu", ""))
        if not index.isdigit():
            continue
        total = str((e.get("memory") or {}).get("total", ""))
        pci = e.get("pci") or {}
        devices.append(
            _Device(
                vendor="rebellions",
                index=int(index),
                product=str(e.get("name", "")),
                memory_bytes=int(total) if total.isdigit() and int(total) > 0 else None,
                pci_bdf=str(pci.get("bus_id", "")),
                link_speed=str(pci.get("link_speed", "")),
                source="rbln-stat",
                extra={"firmware": str(e.get("fw_ver", ""))},
            )
        )
    return devices


# --- existing detection -------------------------------------------------------------------


def _from_hardware_info(info: HardwareInfo) -> _Device:
    vendor = getattr(info.vendor, "value", info.vendor)
    return _Device(
        vendor=str(vendor),
        index=int(info.device_id),
        product=str(info.name),
        memory_bytes=info.memory_total or None,
        source="hardware_detector",
    )


def _detector_devices(detector: HardwareDetector, vendor: str) -> list[_Device]:
    try:
        return [_from_hardware_info(i) for i in detector.get_by_vendor(vendor)]
    except Exception as e:  # discovery must not raise for absent or broken SDKs
        logger.debug(f"HardwareDetector({vendor}) failed: {e}")
        return []


# --- conversion ---------------------------------------------------------------------------


def _model(dev: _Device) -> str:
    if dev.vendor == "rebellions":
        model = _REBELLIONS_MODELS.get(dev.product, dev.product)
        if model == "Rebellions ATOM":  # name used by HardwareDetector
            model = "ATOM"
    elif dev.vendor == "furiosa":
        model = "RNGD" if "RNGD" in dev.product.upper() else dev.product
    else:
        model = re.sub(r"^(NVIDIA|AMD)\s+", "", dev.product.strip())
    # Must be usable in the node label key airm.ai-computing.io/<vendor>.<model>.
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-_.") or "unknown"


def _memory_kind(vendor: str, model: str, spec: dict[str, Any]) -> str:
    if "memory_kind" in spec:
        return spec["memory_kind"]
    if vendor == "nvidia":
        return "hbm" if _NVIDIA_HBM.search(model.replace("-", " ")) else "gddr"
    if vendor == "amd":
        return "hbm" if model.upper().startswith(("MI", "INSTINCT")) else "gddr"
    return "dram"


def _to_accelerator(dev: _Device, node: str) -> Accelerator | None:
    model = _model(dev)
    spec = _SPECS.get((dev.vendor, model), {})
    memory = dev.memory_bytes or spec.get("memory_bytes")
    if not memory:
        logger.warning(
            f"{dev.vendor}:{dev.index} ({model}): memory size unknown, not reported"
        )
        return None
    extra = {"product": dev.product, "source": dev.source}
    if dev.memory_bytes is None:
        extra["memory_source"] = "datasheet"
    if dev.pci_bdf:
        extra["pci_bdf"] = dev.pci_bdf
    extra.update({k: v for k, v in dev.extra.items() if v})
    return Accelerator(
        id=f"{node}/{dev.vendor}:{dev.index}",
        node=node,
        vendor=dev.vendor,
        model=model,
        memory_bytes=int(memory),
        memory_kind=_memory_kind(dev.vendor, model, spec),
        peak_tflops=dict(spec.get("peak_tflops", {})),
        peak_source=spec.get("peak_source", ""),
        power_limit_w=dev.power_limit_w or None,
        interconnect=_pcie(dev.link_speed, dev.pci_bdf),
        extra=extra,
    ).validate()


class RmAbstractProvider:
    """
    AcceleratorProvider (rm_core) backed by rm_abstract device detection.

    Args:
        node: Node name used in accelerator ids. Default: AIRM_NODE_NAME, else hostname.
        occupancy: Report per-card holders (``extra``). Default: AIRM_OCCUPANCY, else on.

    Example:
        from rm_core import registry
        provider = registry.load("airm.accelerator_providers", "rm-abstract")
        for acc in provider.discover():
            print(acc.id, acc.model, acc.memory_bytes)
    """

    name = "rm-abstract"

    # vendor -> management tool probe; vendors without a tool use HardwareDetector only
    _TOOLS = {"nvidia": _nvidia_smi, "furiosa": _furiosa_smi, "rebellions": _rbln_stat}
    _VENDORS = ("nvidia", "amd", "furiosa", "rebellions")

    def __init__(self, node: str | None = None, occupancy: bool | None = None):
        self.node = node or node_name()
        if occupancy is None:
            occupancy = os.environ.get("AIRM_OCCUPANCY", "1").lower() not in (
                "0",
                "false",
                "no",
            )
        self.occupancy = occupancy
        self._detector = HardwareDetector()
        self._warned = False

    def _devices(self, vendor: str) -> list[_Device]:
        probe = self._TOOLS.get(vendor)
        if probe is not None:
            try:
                found = probe()
            except Exception as e:  # a malformed tool output must not break discovery
                logger.debug(f"{vendor} tool probe failed: {e}")
                found = []
            if found:
                return found
        self._detector.clear_cache()
        return _detector_devices(self._detector, vendor)

    def discover(self) -> list[Accelerator]:
        """Accelerators on this node, sorted by id; [] when there are none."""
        found: dict[str, Accelerator] = {}
        for vendor in self._VENDORS:
            devices = self._devices(vendor)
            # On ROCm, torch reports AMD GPUs through the CUDA API as well.
            if vendor == "nvidia" and all(
                d.source == "hardware_detector" for d in devices
            ):
                if _detector_devices(self._detector, "amd"):
                    devices = []
            for dev in devices:
                try:
                    acc = _to_accelerator(dev, self.node)
                except Exception as e:
                    logger.warning(f"{dev.vendor}:{dev.index}: skipped ({e})")
                    continue
                if acc is not None:
                    found.setdefault(acc.id, acc)
        accelerators = [found[k] for k in sorted(found)]
        if self.occupancy and accelerators:
            accelerators = self._with_occupancy(accelerators)
        return accelerators

    def _with_occupancy(self, accelerators: list[Accelerator]) -> list[Accelerator]:
        """Add inUse/holders (and busy for host-held cards) to each accelerator's extra."""
        try:
            held = occupancy.holders(_run)
        except Exception as e:  # occupancy must never break discovery
            logger.warning(f"occupancy not reported: {e}")
            return accelerators
        if held is None:
            if not self._warned:
                logger.warning(
                    "host processes are not visible (run with hostPID): occupancy not reported"
                )
                self._warned = True
            return accelerators
        out = []
        for acc in accelerators:
            index = int(acc.id.rsplit(":", 1)[1])
            extra = occupancy.occupancy_extra(held.get((acc.vendor, index), []))
            out.append(replace(acc, extra={**acc.extra, **extra}))
        return out

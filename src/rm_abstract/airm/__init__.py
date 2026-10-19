"""
ai-rm protocol v0.1 adapters.

- RmAbstractProvider: rm_core AcceleratorProvider over this package's device detection
  (entry point group ``airm.accelerator_providers``, name ``rm-abstract``)
- daemon: node daemon that publishes AcceleratorProfiles and node labels
  (``python -m rm_abstract.airm.daemon``)
"""

from .provider import RmAbstractProvider, node_name

__all__ = ["RmAbstractProvider", "node_name"]

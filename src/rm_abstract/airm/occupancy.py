"""
Per-card occupancy: which processes hold each accelerator, and whether they run in a Kubernetes pod.

A device plugin only knows about the cards it handed to pods. A process started directly on the
host (outside Kubernetes) can hold a card that the plugin still counts as free, so a pod placed
there fails with EBUSY. This module finds the holders of every card, read-only:

1. The vendor tools: ``furiosa-smi ps --format json`` (RNGD; ``dev_name`` is ``npu<N>:[cores]``),
   the Rebellions driver's per-device context list ``/sys/class/rebellions/rbln<N>/ctx_info``
   (what ``rbln-stat`` prints) and ``rbln-stat --json`` ``contexts`` where their ``npu`` index
   maps to exactly one device (``npu`` restarts at 0 in each device group).
2. Open file descriptors in ``/proc/<pid>/fd`` on the cards' compute device nodes
   (``/dev/rngd/npu<N>ch*|pe*``, ``/dev/rbln<N>``), which catches holders a tool version misses.

Each holder is classified by ``/proc/<pid>/cgroup``: ``kubepods`` in the path means a pod
(``managed: k8s``), anything else is a host process (``managed: host``). Only the executable's
basename is kept, never arguments, users or hostnames.

Both sources need the host's PID namespace (DaemonSet ``hostPID: true``). Without it the daemon
only sees its own container; ``host_pids_visible()`` detects that, and occupancy is then not
reported at all rather than reported as free.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PROC = "/proc"
RBLN_SYSFS = "/sys/class/rebellions"

K8S = "k8s"
HOST = "host"

# Monitoring tools open device nodes for a moment; they never hold a card.
_MONITORS = {"furiosa-smi", "rbln-stat", "nvidia-smi"}
# RNGD compute channels and PE groups (management/BAR nodes are opened by exporters as well).
_FD_PATTERNS = (
    ("furiosa", re.compile(r"^/dev/rngd/npu(\d+)(?:ch|pe)")),
    ("rebellions", re.compile(r"^/dev/rbln(\d+)$")),
)

Run = Callable[[list[str]], "str | None"]
Pids = dict[tuple[str, int], set[int]]


@dataclass(frozen=True)
class Holder:
    pid: int
    command: str  # executable basename
    managed: str  # "k8s" or "host"

    def as_dict(self) -> dict[str, object]:
        return {"pid": self.pid, "command": self.command, "managed": self.managed}


def _add(pids: Pids, vendor: str, index: int, pid: int) -> None:
    if pid > 0:
        pids.setdefault((vendor, index), set()).add(pid)


# --- sources ------------------------------------------------------------------------------


def furiosa_pids(run: Run) -> Pids:
    """(furiosa, index) -> pids from ``furiosa-smi ps --format json``."""
    pids: Pids = {}
    out = run(["furiosa-smi", "ps", "--format", "json"])
    try:
        entries = json.loads(out) if out else []
    except ValueError:
        return pids
    for e in entries if isinstance(entries, list) else []:
        m = re.match(r"^npu(\d+)", str(e.get("dev_name", "")))
        if m and str(e.get("pid", "")).isdigit():
            _add(pids, "furiosa", int(m.group(1)), int(e["pid"]))
    return pids


def rebellions_pids(run: Run, sysfs: str = RBLN_SYSFS) -> Pids:
    """(rebellions, index) -> pids from the driver's ctx_info and ``rbln-stat --json`` contexts."""
    pids: Pids = {}
    try:
        names = os.listdir(sysfs)
    except OSError:
        names = []
    for name in names:
        m = re.fullmatch(r"rbln(\d+)", name)
        if not m:
            continue
        try:
            with open(os.path.join(sysfs, name, "ctx_info")) as f:
                lines = f.readlines()
        except OSError:
            continue
        # one context per line: pid, ctx_id, priority, ptid, memalloc, status, util (pid 0 = killed)
        for line in lines:
            pid = line.split(",", 1)[0].strip()
            if pid.isdigit():
                _add(pids, "rebellions", int(m.group(1)), int(pid))

    out = run(["rbln-stat", "--json"])
    try:
        stat = json.loads(out) if out else {}
        devices, contexts = stat.get("devices") or [], stat.get("contexts") or []
    except (ValueError, AttributeError):
        return pids
    by_npu: dict[tuple[str, str], list[int]] = {}
    for d in devices:
        m = re.fullmatch(r"rbln(\d+)", str(d.get("device", "")))
        if m:
            key = (str(d.get("group_id", "")), str(d.get("npu", "")))
            by_npu.setdefault(key, []).append(int(m.group(1)))
    for c in contexts:
        index = by_npu.get((str(c.get("group_id", "")), str(c.get("npu", ""))), [])
        if len(index) == 1 and str(c.get("pid", "")).isdigit():
            _add(pids, "rebellions", index[0], int(c["pid"]))
    return pids


def fd_pids(proc: str = PROC) -> Pids:
    """(vendor, index) -> pids with a card's compute device node open."""
    pids: Pids = {}
    try:
        entries = os.listdir(proc)
    except OSError:
        return pids
    for pid in entries:
        if not pid.isdigit():
            continue
        fd_dir = os.path.join(proc, pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue  # gone, or not ours to read
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            for vendor, pattern in _FD_PATTERNS:
                m = pattern.match(target)
                if m:
                    _add(pids, vendor, int(m.group(1)), int(pid))
    return pids


# --- classification -----------------------------------------------------------------------


def host_pids_visible(proc: str = PROC) -> bool:
    """True when /proc is the host's: PID 1 is the host init, not a container's first process."""
    try:
        with open(os.path.join(proc, "1", "cgroup")) as f:
            return "kubepods" not in f.read()
    except OSError:
        return False


def holder(pid: int, proc: str = PROC) -> Holder | None:
    """The holder behind a pid, or None when the process is gone or is a monitoring tool."""
    base = os.path.join(proc, str(pid))
    try:
        with open(os.path.join(base, "cmdline"), "rb") as f:
            argv0 = f.read().split(b"\0", 1)[0].decode("utf-8", "replace")
    except OSError:
        return None
    command = os.path.basename(argv0)
    if not command:
        try:
            with open(os.path.join(base, "comm")) as f:
                command = f.read().strip()
        except OSError:
            return None
    if command in _MONITORS:
        return None
    try:
        with open(os.path.join(base, "cgroup")) as f:
            cgroup = f.read()
    except OSError:
        cgroup = (
            ""  # unreadable: count it as a host process, so the card is not offered
        )
    return Holder(pid, command, K8S if "kubepods" in cgroup else HOST)


def holders(
    run: Run, proc: str | None = None, sysfs: str | None = None
) -> dict[tuple[str, int], list[Holder]] | None:
    """(vendor, index) -> holders of each busy card; None when host processes are not visible."""
    proc = proc or PROC
    sysfs = sysfs or RBLN_SYSFS
    if not host_pids_visible(proc):
        return None
    merged: Pids = {}
    for source in (
        lambda: furiosa_pids(run),
        lambda: rebellions_pids(run, sysfs),
        lambda: fd_pids(proc),
    ):
        try:
            found = source()
        except Exception as e:  # one broken source must not hide the others
            logger.debug(f"occupancy source failed: {e}")
            continue
        for key, pids in found.items():
            merged.setdefault(key, set()).update(pids)
    own = os.getpid()
    out = {}
    for key, pids in merged.items():
        hs = [h for h in (holder(p, proc) for p in sorted(pids) if p != own) if h]
        if hs:
            out[key] = hs
    return out


def occupancy_extra(card_holders: list[Holder]) -> dict[str, str]:
    """AcceleratorProfile ``spec.extra`` keys for one card.

    ``inUse`` and ``holders`` (JSON list of {pid, command, managed}) always; a card held by a host
    process also gets ``health: busy`` and ``healthy: "false"``, which rm-controller's placement
    honours (it skips cards with ``healthy`` false). Cards held only by pods stay available: the
    device plugin already accounts for them.
    """
    extra = {
        "inUse": "true" if card_holders else "false",
        "holders": json.dumps(
            [h.as_dict() for h in card_holders], separators=(",", ":")
        ),
    }
    if any(h.managed == HOST for h in card_holders):
        extra.update(health="busy", healthy="false")
    return extra


def host_held(extra: dict[str, str]) -> bool:
    return extra.get("health") == "busy"

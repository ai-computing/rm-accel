"""
ai-rm node daemon: publish this node's accelerators to Kubernetes (PROTOCOL.md section 3.1).

Every interval it runs AcceleratorProvider.discover(), applies one cluster-scoped
AcceleratorProfile per device (rm_core.k8s.accelerator_profile), removes profiles of devices that
left the node (missing for AIRM_REMOVE_AFTER discoveries in a row), and patches the summary
node labels (rm_core.k8s.node_labels). It also serves the ``airm_accelerators{vendor,model}``
gauge on /metrics.

    python -m rm_abstract.airm.daemon --once --dry-run      # print the objects, touch nothing
    python -m rm_abstract.airm.daemon                       # in-cluster (ServiceAccount)

Applying needs the Kubernetes client: ``pip install "rm-abstract[k8s]"``.

Environment:
    AIRM_NODE_NAME          node name (DaemonSet: spec.nodeName); default hostname
    AIRM_PROVIDER           provider entry point name; default rm-abstract
    AIRM_INTERVAL_S         seconds between discoveries; default 60
    AIRM_METRICS_PORT       /metrics port, 0 disables; default 9400
    AIRM_REMOVE_AFTER       discoveries a device may be missing before removal; default 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

from rm_core import Accelerator, registry
from rm_core.accelerator import VENDORS
from rm_core.k8s import GROUP, LABEL_PREFIX, VERSION, accelerator_profile, node_labels
from rm_core.metrics import prometheus_name

from .provider import node_name

logger = logging.getLogger("rm_abstract.airm.daemon")

PROVIDER_GROUP = "airm.accelerator_providers"
PLURAL = "acceleratorprofiles"
COMPONENT = "rm_abstract_layer"


def build_objects(node: str, accelerators: list[Accelerator]) -> dict[str, object]:
    """The AcceleratorProfiles and node labels this node should have."""
    return {
        "node": node,
        "profiles": [accelerator_profile(a) for a in accelerators],
        "labels": node_labels(accelerators),
    }


def as_manifest(objects: dict[str, object]) -> dict[str, object]:
    """A kubectl-style List of what would be applied (for --dry-run)."""
    node_patch = {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": {"name": objects["node"], "labels": objects["labels"]},
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": objects["profiles"] + [node_patch],
    }


def _is_summary_label(key: str) -> bool:
    """airm.ai-computing.io/<vendor>.<model> or .../managed, i.e. labels this daemon owns."""
    if not key.startswith(LABEL_PREFIX):
        return False
    name = key[len(LABEL_PREFIX) :]
    return name == "managed" or ("." in name and name.split(".", 1)[0] in VENDORS)


class KubePublisher:
    """Writes profiles and node labels with the in-cluster ServiceAccount (or kubeconfig)."""

    def __init__(self):
        try:
            from kubernetes import client, config
        except ImportError as e:
            raise SystemExit(
                'the Kubernetes client is not installed: pip install "rm-abstract[k8s]"'
            ) from e
        try:
            config.load_incluster_config()
        except config.ConfigException:
            try:
                config.load_kube_config()
            except (config.ConfigException, OSError) as e:
                raise SystemExit(
                    f"no in-cluster ServiceAccount and no usable kubeconfig ({e}); "
                    "use --dry-run to only print the objects"
                ) from e
        self._custom = client.CustomObjectsApi()
        self._core = client.CoreV1Api()
        self._api_error = client.exceptions.ApiException

    def publish(self, objects: dict[str, object]) -> None:
        node = objects["node"]
        keep = set()
        for obj in objects["profiles"]:
            name = obj["metadata"]["name"]
            keep.add(name)
            try:
                current = self._custom.get_cluster_custom_object(
                    GROUP, VERSION, PLURAL, name
                )
            except self._api_error as e:
                if e.status != 404:
                    raise
                self._custom.create_cluster_custom_object(GROUP, VERSION, PLURAL, obj)
                logger.info(f"created AcceleratorProfile {name}")
                continue
            obj = dict(obj)
            obj["metadata"] = dict(
                obj["metadata"], resourceVersion=current["metadata"]["resourceVersion"]
            )
            self._custom.replace_cluster_custom_object(
                GROUP, VERSION, PLURAL, name, obj
            )

        listed = self._custom.list_cluster_custom_object(
            GROUP, VERSION, PLURAL, label_selector=f"{LABEL_PREFIX}node={node}"
        )
        for item in listed.get("items", []):
            name = item["metadata"]["name"]
            if name not in keep:
                self._custom.delete_cluster_custom_object(GROUP, VERSION, PLURAL, name)
                logger.info(f"deleted AcceleratorProfile {name} (device gone)")

        labels: dict[str, str | None] = dict(objects["labels"])
        current_labels = self._core.read_node(node).metadata.labels or {}
        for key in current_labels:
            if _is_summary_label(key) and key not in labels:
                labels[key] = None  # merge patch: remove
        self._core.patch_node(node, {"metadata": {"labels": labels}})


class _Retention:
    """Keeps reporting a device that vanished until it has been missing for `cycles` discoveries.

    A vendor tool that times out once makes discover() return no devices for that vendor;
    without this, its profiles and node label would be deleted and recreated, flapping placement.
    """

    def __init__(self, cycles: int):
        self.cycles = max(1, cycles)
        self._last: dict[str, Accelerator] = {}
        self._missing: dict[str, int] = {}

    def apply(self, accelerators: list[Accelerator]) -> list[Accelerator]:
        current = {a.id: a for a in accelerators}
        for acc_id, acc in self._last.items():
            if acc_id in current:
                self._missing.pop(acc_id, None)
                continue
            missing = self._missing.get(acc_id, 0) + 1
            if missing < self.cycles:
                self._missing[acc_id] = missing
                current[acc_id] = acc
            else:
                self._missing.pop(acc_id, None)
                logger.info(f"{acc_id} missing for {missing} discoveries, removing")
        self._last = current
        return [current[k] for k in sorted(current)]


class _Metrics:
    """Serves airm_accelerators{component,vendor,model} from the last discovery."""

    def __init__(self):
        self._text = ""
        self._lock = threading.Lock()

    def update(self, accelerators: list[Accelerator]) -> None:
        name = prometheus_name("accelerators")
        lines = [
            f"# HELP {name} accelerators on this node",
            f"# TYPE {name} gauge",
        ]
        for (vendor, model), n in sorted(
            Counter((a.vendor, a.model) for a in accelerators).items()
        ):
            lines.append(
                f'{name}{{component="{COMPONENT}",vendor="{vendor}",model="{model}"}} {n}'
            )
        with self._lock:
            self._text = "\n".join(lines) + "\n"

    def serve(self, port: int) -> HTTPServer:
        metrics = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split("?", 1)[0] != "/metrics":
                    self.send_error(404)
                    return
                with metrics._lock:
                    body = metrics._text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(
        prog="python -m rm_abstract.airm.daemon",
        description="Publish this node's accelerators as AcceleratorProfiles and node labels.",
    )
    p.add_argument(
        "--provider",
        default=env("AIRM_PROVIDER", "rm-abstract"),
        help=f"entry point name in {PROVIDER_GROUP} (default: %(default)s)",
    )
    p.add_argument(
        "--node",
        default=None,
        help="node name (default: AIRM_NODE_NAME, else hostname)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=float(env("AIRM_INTERVAL_S", "60")),
        help="seconds between discoveries (default: %(default)s)",
    )
    p.add_argument(
        "--metrics-port",
        type=int,
        default=int(env("AIRM_METRICS_PORT", "9400")),
        help="port for /metrics, 0 disables (default: %(default)s)",
    )
    p.add_argument(
        "--remove-after",
        type=int,
        default=int(env("AIRM_REMOVE_AFTER", "3")),
        help="remove a device's profile after it is missing this many discoveries in a row "
        "(default: %(default)s)",
    )
    p.add_argument("--once", action="store_true", help="run one discovery and exit")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the objects that would be applied instead of applying them",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.node:
        os.environ["AIRM_NODE_NAME"] = (
            args.node
        )  # providers read the node name from here
    provider = registry.load(PROVIDER_GROUP, args.provider)
    node = getattr(provider, "node", None) or node_name()
    publisher = None if args.dry_run else KubePublisher()
    retention = _Retention(args.remove_after)
    metrics = _Metrics()
    if not args.once and args.metrics_port:
        metrics.serve(args.metrics_port)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    while True:
        try:
            accelerators = retention.apply(provider.discover())
            objects = build_objects(node, accelerators)
            metrics.update(accelerators)
            if publisher is None:
                print(json.dumps(as_manifest(objects), indent=2), flush=True)
            else:
                publisher.publish(objects)
                logger.info(f"{node}: {len(accelerators)} accelerators published")
        except Exception as e:
            if args.once:
                logger.error(f"discovery/publish failed: {e}")
                return 1
            logger.exception(
                f"discovery/publish failed, retrying in {args.interval:g}s"
            )
        if args.once or stop.wait(args.interval):
            return 0


if __name__ == "__main__":
    sys.exit(main())

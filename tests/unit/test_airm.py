"""ai-rm protocol v0.1 adapter tests (rm_abstract.airm), no accelerators needed"""

import json

import pytest

rm_core = pytest.importorskip("rm_core")

from rm_core.conformance import check_accelerator_provider  # noqa: E402

from rm_abstract.airm import daemon, provider  # noqa: E402
from rm_abstract.airm.provider import RmAbstractProvider  # noqa: E402
from rm_abstract.common.hardware_detector import (  # noqa: E402
    HardwareInfo,
    HardwareType,
    HardwareVendor,
)

RBLN_STAT = {
    "devices": [
        {
            "npu": 0,
            "name": "RBLN-CA22",
            "fw_ver": "3.0.0",
            "pci": {"bus_id": "0000:83:00.0", "link_speed": "32.0GT/s"},
            "memory": {"used": "0", "total": "16877879296"},
        },
        {
            "npu": 1,
            "name": "RBLN-CA22",
            "fw_ver": "3.0.0",
            "pci": {"bus_id": "0000:84:00.0", "link_speed": "16.0GT/s"},
            "memory": {"used": "0", "total": "16877879296"},
        },
    ]
}
FURIOSA_SMI = [
    {"arch": "RNGD", "dev_name": "npu0", "power_limit": "300 W", "pci_bdf": ""},
    {"arch": "RNGD", "dev_name": "npu3", "power_limit": "300 W", "pci_bdf": ""},
]
NVIDIA_SMI = "0, NVIDIA A100-SXM4-80GB, 81920, 400.00, 00000000:3B:00.0\n"


def fake_tools(outputs):
    """_run replacement: tool name -> stdout (missing tools return None)."""
    return lambda cmd: outputs.get(cmd[0])


@pytest.fixture
def no_hardware(monkeypatch):
    monkeypatch.setenv("AIRM_NODE_NAME", "test-node")
    monkeypatch.setattr(provider, "_run", fake_tools({}))
    monkeypatch.setattr(
        provider.HardwareDetector, "get_by_vendor", lambda self, vendor: []
    )
    return monkeypatch


class TestRmAbstractProvider:
    """AcceleratorProvider contract"""

    def test_no_devices_returns_empty_and_conforms(self, no_hardware):
        p = RmAbstractProvider()
        assert p.discover() == []
        assert check_accelerator_provider(p) == []

    def test_registered_entry_point(self):
        from rm_core import registry

        assert registry.available("airm.accelerator_providers").get("rm-abstract") == (
            "rm_abstract.airm:RmAbstractProvider"
        )

    def test_rebellions_from_rbln_stat(self, no_hardware):
        no_hardware.setattr(
            provider, "_run", fake_tools({"rbln-stat": json.dumps(RBLN_STAT)})
        )
        accs = check_accelerator_provider(RmAbstractProvider())
        assert [a.id for a in accs] == [
            "test-node/rebellions:0",
            "test-node/rebellions:1",
        ]
        a = accs[0]
        assert (a.vendor, a.model, a.memory_kind) == ("rebellions", "ATOM", "gddr")
        assert a.memory_bytes == 16877879296
        assert a.peak_tflops and a.peak_source
        assert a.extra["product"] == "RBLN-CA22"
        assert [x.interconnect for x in accs] == [["pcie5"], ["pcie4"]]

    def test_furiosa_keeps_device_index(self, no_hardware):
        no_hardware.setattr(
            provider, "_run", fake_tools({"furiosa-smi": json.dumps(FURIOSA_SMI)})
        )
        accs = check_accelerator_provider(RmAbstractProvider())
        assert [a.id for a in accs] == ["test-node/furiosa:0", "test-node/furiosa:3"]
        assert all(a.model == "RNGD" and a.memory_kind == "hbm" for a in accs)
        assert accs[0].power_limit_w == 300.0
        assert accs[0].extra["memory_source"] == "datasheet"

    def test_nvidia_from_nvidia_smi(self, no_hardware):
        no_hardware.setattr(provider, "_run", fake_tools({"nvidia-smi": NVIDIA_SMI}))
        (a,) = check_accelerator_provider(RmAbstractProvider())
        assert a.id == "test-node/nvidia:0"
        assert a.model == "A100-SXM4-80GB"
        assert a.memory_bytes == 81920 << 20
        assert a.memory_kind == "hbm"
        assert a.peak_tflops == {} and a.peak_source == ""  # unknown -> omitted
        assert a.extra["pci_bdf"] == "0000:3b:00.0"

    def test_falls_back_to_hardware_detector(self, no_hardware):
        info = HardwareInfo(
            hardware_type=HardwareType.NPU,
            vendor=HardwareVendor.REBELLIONS,
            device_id=0,
            name="Rebellions ATOM",
        )
        no_hardware.setattr(
            provider.HardwareDetector,
            "get_by_vendor",
            lambda self, vendor: [info] if vendor == "rebellions" else [],
        )
        (a,) = check_accelerator_provider(RmAbstractProvider())
        assert (a.id, a.model, a.memory_bytes) == (
            "test-node/rebellions:0",
            "ATOM",
            16 << 30,
        )
        assert a.extra["source"] == "hardware_detector"

    def test_device_without_memory_is_skipped(self, no_hardware):
        info = HardwareInfo(
            hardware_type=HardwareType.GPU,
            vendor=HardwareVendor.NVIDIA,
            device_id=0,
            name="NVIDIA Mystery",
        )
        no_hardware.setattr(
            provider.HardwareDetector,
            "get_by_vendor",
            lambda self, vendor: [info] if vendor == "nvidia" else [],
        )
        assert RmAbstractProvider().discover() == []

    def test_broken_tool_output_does_not_raise(self, no_hardware):
        no_hardware.setattr(
            provider, "_run", fake_tools({"rbln-stat": "not json", "furiosa-smi": "{}"})
        )
        assert RmAbstractProvider().discover() == []


class TestNodeName:
    """Accelerator id node segment"""

    def test_env_overrides_hostname(self, monkeypatch):
        monkeypatch.setenv("AIRM_NODE_NAME", " Worker 01.Example")
        assert provider.node_name() == "worker-01.example"

    def test_hostname_fallback(self, monkeypatch):
        monkeypatch.delenv("AIRM_NODE_NAME", raising=False)
        monkeypatch.setattr(provider.socket, "gethostname", lambda: "Host-A")
        assert provider.node_name() == "host-a"


class TestDaemon:
    """Node daemon objects (no cluster access)"""

    def test_dry_run_prints_profiles_and_node_labels(self, monkeypatch, capsys):
        monkeypatch.setenv("AIRM_NODE_NAME", "sim-node")
        assert daemon.main(["--provider", "sim", "--once", "--dry-run"]) == 0
        manifest = json.loads(capsys.readouterr().out)
        kinds = [i["kind"] for i in manifest["items"]]
        assert kinds == ["AcceleratorProfile", "AcceleratorProfile", "Node"]
        node = manifest["items"][-1]
        assert node["metadata"]["name"] == "sim-node"
        assert node["metadata"]["labels"] == {
            "airm.ai-computing.io/managed": "true",
            "airm.ai-computing.io/furiosa.RNGD": "1",
            "airm.ai-computing.io/rebellions.ATOM": "1",
        }

    def test_dry_run_without_devices(self, no_hardware, capsys):
        assert daemon.main(["--once", "--dry-run"]) == 0
        items = json.loads(capsys.readouterr().out)["items"]
        assert items == [
            {
                "apiVersion": "v1",
                "kind": "Node",
                "metadata": {
                    "name": "test-node",
                    "labels": {"airm.ai-computing.io/managed": "true"},
                },
            }
        ]

    def test_summary_label_ownership(self):
        assert daemon._is_summary_label("airm.ai-computing.io/furiosa.RNGD")
        assert daemon._is_summary_label("airm.ai-computing.io/managed")
        assert not daemon._is_summary_label("airm.ai-computing.io/node")
        assert not daemon._is_summary_label("airm.ai-computing.io/scrape")
        assert not daemon._is_summary_label("kubernetes.io/hostname")

    def test_metrics_text(self):
        from rm_core.reference import SimAcceleratorProvider

        m = daemon._Metrics()
        m.update(SimAcceleratorProvider("n1").discover())
        assert (
            'airm_accelerators{component="rm_abstract_layer",vendor="furiosa",model="RNGD"} 1'
            in m._text
        )


class FakeApiError(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


class FakeCustomObjects:
    def __init__(self, objects):
        self.objects = {o["metadata"]["name"]: o for o in objects}

    def get_cluster_custom_object(self, group, version, plural, name):
        if name not in self.objects:
            raise FakeApiError(404)
        return self.objects[name]

    def create_cluster_custom_object(self, group, version, plural, body):
        body = dict(body, metadata=dict(body["metadata"], resourceVersion="1"))
        self.objects[body["metadata"]["name"]] = body

    def replace_cluster_custom_object(self, group, version, plural, name, body):
        assert (
            body["metadata"]["resourceVersion"]
            == self.objects[name]["metadata"]["resourceVersion"]
        )
        self.objects[name] = body

    def list_cluster_custom_object(self, group, version, plural, label_selector):
        key, value = label_selector.split("=")
        items = [
            o
            for o in self.objects.values()
            if o["metadata"]["labels"].get(key) == value
        ]
        return {"items": items}

    def delete_cluster_custom_object(self, group, version, plural, name):
        del self.objects[name]


class FakeCore:
    def __init__(self, labels):
        self.labels = labels
        self.patches = []

    def read_node(self, name):
        from types import SimpleNamespace

        return SimpleNamespace(metadata=SimpleNamespace(labels=dict(self.labels)))

    def patch_node(self, name, body):
        self.patches.append((name, body))


class TestKubePublisher:
    """Apply logic against fake Kubernetes APIs"""

    def test_publish_creates_replaces_and_prunes(self):
        from rm_core.reference import SimAcceleratorProvider

        accs = SimAcceleratorProvider("n1").discover()
        objects = daemon.build_objects("n1", accs)
        existing = dict(objects["profiles"][0])
        existing["metadata"] = dict(existing["metadata"], resourceVersion="7")
        stale = {
            "metadata": {
                "name": "n1.nvidia-0",
                "labels": {"airm.ai-computing.io/node": "n1"},
                "resourceVersion": "3",
            }
        }
        other_node = {
            "metadata": {
                "name": "n2.nvidia-0",
                "labels": {"airm.ai-computing.io/node": "n2"},
                "resourceVersion": "3",
            }
        }
        custom = FakeCustomObjects([existing, stale, other_node])
        core = FakeCore(
            {
                "kubernetes.io/hostname": "n1",
                "airm.ai-computing.io/managed": "true",
                "airm.ai-computing.io/nvidia.A100": "1",
            }
        )
        pub = daemon.KubePublisher.__new__(daemon.KubePublisher)
        pub._custom, pub._core, pub._api_error = custom, core, FakeApiError

        pub.publish(objects)

        assert sorted(custom.objects) == sorted(
            [p["metadata"]["name"] for p in objects["profiles"]] + ["n2.nvidia-0"]
        )
        ((name, body),) = core.patches
        assert name == "n1"
        assert body["metadata"]["labels"] == {
            "airm.ai-computing.io/managed": "true",
            "airm.ai-computing.io/furiosa.RNGD": "1",
            "airm.ai-computing.io/rebellions.ATOM": "1",
            "airm.ai-computing.io/nvidia.A100": None,
        }


class TestRetention:
    """A device missing from one discovery is not removed right away"""

    def test_device_removed_after_consecutive_misses(self):
        from rm_core.reference import SimAcceleratorProvider

        accs = SimAcceleratorProvider("n1").discover()
        r = daemon._Retention(3)
        ids = lambda xs: [a.id for a in xs]  # noqa: E731
        assert ids(r.apply(accs)) == ids(accs)
        assert ids(r.apply(accs[:1])) == ids(accs)  # missing once: kept
        assert ids(r.apply(accs)) == ids(accs)  # back: counter reset
        assert ids(r.apply(accs[:1])) == ids(accs)
        assert ids(r.apply(accs[:1])) == ids(accs)
        assert ids(r.apply(accs[:1])) == ids(accs[:1])  # third miss in a row: removed
        assert ids(r.apply([])) == ids(accs[:1])

    def test_once_reports_discovery_as_is(self):
        assert daemon._Retention(1).apply([]) == []

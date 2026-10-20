"""Per-card occupancy (rm_abstract.airm.occupancy) from recorded tool output, no accelerators needed.

Fixtures in tests/fixtures/occupancy (ids, serials and argument paths replaced by placeholders):
- furiosa-smi-ps.host-held.json: ``furiosa-smi ps --format json`` (furiosa-smi 2026.4.0) on a 2-card
  RNGD node where two training processes started on the host held both cards (ai-rm#32)
- furiosa-smi-ps.idle.json, furiosa-smi-info.json: the same tool on idle cards / ``info --format json``
- rbln-stat.idle.json: ``rbln-stat --json`` (3.0.0) on a 4-card ATOM host, no contexts
- rbln-stat.contexts.json: the same with ``contexts`` entries shaped like rbln-stat 3.0.0's
  ``make_json_ctx_data`` (no ATOM was in use while recording)
- cgroup.*: ``/proc/<pid>/cgroup`` of a host process, a pod container and the host init (cgroup v2)
"""

import json
import os
from pathlib import Path

import pytest

rm_core = pytest.importorskip("rm_core")

from rm_abstract.airm import daemon, occupancy, provider  # noqa: E402
from rm_abstract.airm.occupancy import HOST, K8S, Holder  # noqa: E402
from rm_abstract.airm.provider import RmAbstractProvider  # noqa: E402

FIX = Path(__file__).parent.parent / "fixtures" / "occupancy"


def fixture(name):
    return (FIX / name).read_text(encoding="utf-8")


def fake_tools(outputs):
    """_run replacement keyed by the tool and its subcommand, e.g. "furiosa-smi ps"."""
    return lambda cmd: outputs.get(" ".join(cmd[:2]))


def make_proc(root, init="cgroup.init", procs=()):
    """A fake /proc: PID 1 with the `init` cgroup, then (pid, argv0, cgroup fixture, fd targets)."""
    for pid, argv0, cgroup, fds in [(1, "/sbin/init", init, [])] + list(procs):
        d = root / str(pid)
        (d / "fd").mkdir(parents=True)
        (d / "cmdline").write_bytes(argv0.encode() + b"\0--flag\0")
        (d / "cgroup").write_text(fixture(cgroup))
        for i, target in enumerate(fds):
            os.symlink(target, d / "fd" / str(i))
    return str(root)


@pytest.fixture
def symlinks(tmp_path):
    try:
        os.symlink("x", tmp_path / "probe")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")


class TestSources:
    def test_furiosa_smi_ps(self):
        run = fake_tools({"furiosa-smi ps": fixture("furiosa-smi-ps.host-held.json")})
        assert occupancy.furiosa_pids(run) == {
            ("furiosa", 0): {37839},
            ("furiosa", 1): {38166},
        }

    def test_furiosa_idle_and_broken(self):
        assert occupancy.furiosa_pids(fake_tools({"furiosa-smi ps": "[]"})) == {}
        assert occupancy.furiosa_pids(fake_tools({"furiosa-smi ps": "oops"})) == {}
        assert occupancy.furiosa_pids(fake_tools({})) == {}

    def test_rbln_stat_contexts_only_when_npu_is_unambiguous(self, tmp_path):
        run = fake_tools({"rbln-stat --json": fixture("rbln-stat.contexts.json")})
        # npu 1 is rbln1 alone; npu 0 is shared by rbln0/2/3 and cannot be attributed
        assert occupancy.rebellions_pids(run, str(tmp_path / "none")) == {
            ("rebellions", 1): {4242}
        }

    def test_rbln_sysfs_ctx_info(self, tmp_path):
        ctx = {
            "rbln0": "",
            "rbln3": "4343, 1, 0, 4343, 1048576, 1, 0\n0, 2, 0, 0, 0, 1, 0\n",
        }
        for name, text in ctx.items():
            (tmp_path / name).mkdir()
            (tmp_path / name / "ctx_info").write_text(text)
        (tmp_path / "rsd0").mkdir()
        run = fake_tools({"rbln-stat --json": fixture("rbln-stat.idle.json")})
        assert occupancy.rebellions_pids(run, str(tmp_path)) == {
            ("rebellions", 3): {4343}
        }

    def test_fd_scan(self, tmp_path, symlinks):
        proc = make_proc(
            tmp_path / "proc",
            procs=[
                (
                    100,
                    "/usr/bin/python3",
                    "cgroup.host",
                    ["/dev/rngd/npu1ch0", "/dev/rngd/npu1pe0-3"],
                ),
                (
                    101,
                    "/usr/bin/exporter",
                    "cgroup.k8s",
                    ["/dev/rngd/npu0mgmt"],
                ),  # management only
                (102, "/usr/bin/python3", "cgroup.k8s", ["/dev/rbln2", "/dev/null"]),
            ],
        )
        assert occupancy.fd_pids(proc) == {
            ("furiosa", 1): {100},
            ("rebellions", 2): {102},
        }


class TestClassification:
    def test_host_and_pod(self, tmp_path):
        proc = make_proc(
            tmp_path,
            procs=[
                (37839, "python3", "cgroup.host", []),
                (200, "/usr/local/bin/vllm", "cgroup.k8s", []),
                (300, "/usr/bin/furiosa-smi", "cgroup.host", []),
            ],
        )
        assert occupancy.holder(37839, proc) == Holder(37839, "python3", HOST)
        assert occupancy.holder(200, proc) == Holder(200, "vllm", K8S)
        assert occupancy.holder(300, proc) is None  # monitoring tool
        assert occupancy.holder(999, proc) is None  # gone

    def test_container_without_host_pid_namespace(self, tmp_path):
        proc = make_proc(tmp_path, init="cgroup.k8s")
        assert not occupancy.host_pids_visible(proc)
        assert occupancy.holders(fake_tools({}), proc) is None
        assert not occupancy.host_pids_visible(str(tmp_path / "missing"))

    def test_extra(self):
        assert occupancy.occupancy_extra([]) == {"inUse": "false", "holders": "[]"}
        pod = occupancy.occupancy_extra([Holder(200, "vllm", K8S)])
        assert pod["inUse"] == "true" and "healthy" not in pod and "health" not in pod
        extra = occupancy.occupancy_extra([Holder(37839, "python3", HOST)])
        assert extra["health"] == "busy" and extra["healthy"] == "false"
        assert json.loads(extra["holders"]) == [
            {"pid": 37839, "command": "python3", "managed": "host"}
        ]


@pytest.fixture
def second_rngd_node(tmp_path, monkeypatch):
    """The ai-rm#32 node: 2 RNGD, each held by a training process started on the host."""
    proc = make_proc(
        tmp_path / "proc",
        procs=[
            (37839, "python3", "cgroup.host", []),
            (38166, "python3", "cgroup.host", []),
        ],
    )
    monkeypatch.setenv("AIRM_NODE_NAME", "rngd-node")
    monkeypatch.setenv("AIRM_OCCUPANCY", "1")
    monkeypatch.setattr(occupancy, "PROC", proc)
    monkeypatch.setattr(occupancy, "RBLN_SYSFS", str(tmp_path / "none"))
    monkeypatch.setattr(
        provider,
        "_run",
        fake_tools(
            {
                "furiosa-smi info": fixture("furiosa-smi-info.json"),
                "furiosa-smi ps": fixture("furiosa-smi-ps.host-held.json"),
            }
        ),
    )
    monkeypatch.setattr(provider.HardwareDetector, "get_by_vendor", lambda self, v: [])
    return monkeypatch


class TestProviderAndDaemon:
    def test_host_held_cards_are_busy(self, second_rngd_node):
        accs = RmAbstractProvider().discover()
        assert [a.id for a in accs] == ["rngd-node/furiosa:0", "rngd-node/furiosa:1"]
        for a, pid in zip(accs, (37839, 38166)):
            assert a.extra["inUse"] == "true" and a.extra["healthy"] == "false"
            assert json.loads(a.extra["holders"]) == [
                {"pid": pid, "command": "python3", "managed": "host"}
            ]

    def test_idle_cards_are_free(self, second_rngd_node):
        second_rngd_node.setattr(
            provider,
            "_run",
            fake_tools(
                {
                    "furiosa-smi info": fixture("furiosa-smi-info.json"),
                    "furiosa-smi ps": fixture("furiosa-smi-ps.idle.json"),
                }
            ),
        )
        accs = RmAbstractProvider().discover()
        assert [(a.extra["inUse"], a.extra["holders"]) for a in accs] == [
            ("false", "[]")
        ] * 2
        assert all("healthy" not in a.extra for a in accs)
        assert (
            daemon.build_objects("rngd-node", accs)["labels"][daemon.HOST_HELD_LABEL]
            == "0"
        )

    def test_disabled(self, second_rngd_node):
        second_rngd_node.setenv("AIRM_OCCUPANCY", "0")
        accs = RmAbstractProvider().discover()
        assert all("inUse" not in a.extra for a in accs)
        assert daemon.HOST_HELD_LABEL not in daemon.build_objects("n", accs)["labels"]

    def test_profiles_label_and_metrics(self, second_rngd_node):
        accs = RmAbstractProvider().discover()
        objects = daemon.build_objects("rngd-node", accs)
        assert objects["labels"][daemon.HOST_HELD_LABEL] == "2"
        assert objects["profiles"][0]["spec"]["extra"]["healthy"] == "false"
        m = daemon._Metrics()
        m.update(accs)
        tag = 'component="rm_abstract_layer",vendor="furiosa",model="RNGD"'
        assert f"rm_accel_accelerators_host_held{{{tag}}} 2" in m._text
        assert f"rm_accel_accelerators_in_use{{{tag}}} 2" in m._text
        assert daemon._is_summary_label("airm.ai-computing.io/host-held")

    def test_rm_core_reads_the_profile_back(self, second_rngd_node):
        from rm_core.k8s import from_profile

        acc = RmAbstractProvider().discover()[0]
        back = from_profile(daemon.build_objects("rngd-node", [acc])["profiles"][0])
        assert back.extra["healthy"] == "false" and back.extra["inUse"] == "true"

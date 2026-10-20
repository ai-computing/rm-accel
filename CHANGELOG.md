# Changelog

## 0.5.0 / Chart 0.6.0 — 2026-10-20

Found in the ai-rm end-to-end run (ai-rm#32): two training processes started on a node outside Kubernetes held both
of its RNGD cards, the device plugin and the AcceleratorProfiles reported them free, and a decode pod placed there
failed with `EBUSY` (#9).

### Added
- Card occupancy (`rm_abstract.airm.occupancy`): per card, the holders from `furiosa-smi ps --format json`, the
  Rebellions driver's `ctx_info` / `rbln-stat --json` contexts and `/proc/<pid>/fd`, each classified by
  `/proc/<pid>/cgroup` as `k8s` (pod) or `host`. Published in `spec.extra` as `inUse` and `holders` (pid, executable
  basename, managed). A card held by a host process also gets `health: busy` and `healthy: "false"`, which
  rm-controller already skips. `AIRM_OCCUPANCY=0` turns it off; without the host PID namespace no occupancy is
  published (with a warning).
- Node label `airm.ai-computing.io/host-held=<n>`; `/metrics` series `rm_accel_accelerators_in_use` and
  `rm_accel_accelerators_host_held`.
- Chart values `hostPID` (default `true`) and `occupancy` (default `true`).

## 0.4.2 / Chart 0.5.0 — 2026-10-19

### Added
- Node daemon image `ghcr.io/ai-computing/rm-accel` ([`Dockerfile`](Dockerfile)): python:3.11-slim with
  rm-abstract (no torch), rm-core v0.1.1, the Kubernetes client and numpy. Vendor tools are not baked in.
- Workflow `Image`: builds and smoke-tests the image on pull requests, publishes `:master` from master and
  `:X.Y.Z` / `:X.Y` from `vX.Y.Z` tags (GITHUB_TOKEN, `packages: write`).
- Chart value `hostBin` (default `/usr/bin`): the host's vendor tools mounted read-only at `/host/usr/bin`,
  which is on the image's `PATH`.

### Changed
- Chart default image `ghcr.io/ai-computing/rm-accel` (was the never-published `rm-abstract-node`), tag =
  `appVersion` 0.4.2. Sites no longer need the init-container workaround; the extension points stay.

## Chart 0.4.2 — 2026-10-19

Found while running the daemon on the live cluster (ai-rm end-to-end run).

### Fixed
- Writable `/tmp` under the read-only root: `rbln-stat` (PyInstaller) unpacks itself under `TMPDIR`
  and failed with "Could not create temporary directory", so no ATOM was published.
- `hostDev` (default on) mounts the host's `/dev` live. A privileged container sees `/dev` as it was at
  start; an ATOM that the driver re-created after a pod released it dropped out of discovery.

### Added
- Site extension points, all empty by default: `command`, `extraEnv`, `initContainers`,
  `extraVolumes`, `extraVolumeMounts` (run the daemon from a stock Python image until the
  `rm-abstract-node` image is published).

## 0.4.1 — 2026-10-19

### Fixed
- Rebellions: take the device index from `rbln<N>`, not `npu`. `rbln-stat` restarts `npu` at 0 in each
  device group, so a 4-card host was reported as 2 ATOM (ids collided).

## 0.4.0 — 2026-10-19

Repositioned as the **L1 accelerator abstraction** of [ai-rm](https://github.com/ai-computing/ai-rm).

### Added
- ai-rm protocol v0.1 adapter `rm_abstract.airm`: `AcceleratorProvider` `rm-abstract`
  (entry point `airm.accelerator_providers`), discovery through read-only vendor tools with
  `HardwareDetector` as fallback, datasheet `peak_tflops` for RNGD and ATOM.
- Node daemon `python -m rm_abstract.airm.daemon`: `AcceleratorProfile` per device, node labels
  `airm.ai-computing.io/<vendor>.<model>`, `airm_accelerators` on `/metrics`.
- Helm chart `deploy/helm/rm-abstract-node` (DaemonSet, RBAC).
- `ai-rm.yaml` component manifest; dependency on rm-core v0.1.1.
- `tools/check_sensitive.py` and a CI job that scans every push.
- README in English with a Korean translation (`README.ko.md`).

### Changed
- Package metadata: description, keywords, authors and URLs now point to `ai-computing`.
- Ruff configuration moved to `[tool.ruff.lint]`; code formatted with ruff and black.
- CI runs the full `rm_core.conformance` and a daemon dry-run; mypy is reported but non-blocking
  while existing type errors are fixed.
- `test_all_examples.sh` → `scripts/run_examples.sh` (no hard-coded path);
  `test_multi_engine.py` → `scripts/manual/multi_engine_check.py`.

### Moved
- `CLEANUP_SUMMARY.md`, `docs/FINAL_TEST_REPORT.md`, `docs/TEST_RESULTS.md` and
  `docs/PRESENTATION.md` → `docs/archive/` (point-in-time, not maintained).

## 0.3.0

vLLM-compatible `LLM` API on GPU and CPU; vLLM, Triton and TorchServe engines with automatic server
management; plugin registry for resources and engines.

# Changelog

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

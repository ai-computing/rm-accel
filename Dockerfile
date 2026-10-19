# ai-rm L1 node daemon image (ghcr.io/ai-computing/rm-accel): python -m rm_abstract.airm.daemon
#
# Only what the daemon needs: rm-abstract (without its torch dependency; discovery runs the vendor
# tools), rm-core, the Kubernetes client and numpy. Vendor CLI tools (nvidia-smi, furiosa-smi,
# rbln-stat) are not baked in: they match the host's driver, so the Helm chart mounts the host's
# /usr/bin read-only at /host/usr/bin (value hostBin), which is on PATH here.
#
#   docker build -t ghcr.io/ai-computing/rm-accel:dev .
#   docker run --rm ghcr.io/ai-computing/rm-accel:dev --provider sim --once --dry-run
FROM python:3.11-slim

ARG RM_CORE_REF=v0.1.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/usr/local/bin:/usr/bin:/bin:/host/usr/bin

COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN pip install --no-cache-dir --no-deps /src \
        "https://github.com/ai-computing/rm-core/archive/refs/tags/${RM_CORE_REF}.tar.gz" \
    && pip install --no-cache-dir "kubernetes>=24.2.0" "numpy>=1.20,<3" \
    && rm -rf /src \
    && python -m rm_abstract.airm.daemon --provider sim --once --dry-run >/dev/null

# Root: the vendor tools read the device nodes (the chart runs the container privileged).
ENTRYPOINT ["python", "-m", "rm_abstract.airm.daemon"]

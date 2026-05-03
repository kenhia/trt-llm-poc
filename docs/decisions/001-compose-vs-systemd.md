# Decision 001: Docker Compose vs systemd for Autostart

**Date**: 2026-05-01  
**Status**: Decided  
**Research ref**: R-004 (research.md)

## Context

The serving container needs to start automatically after a host reboot (FR-005, SC-003).
Two standard approaches on Linux are Docker Compose with `restart: unless-stopped` and a
systemd unit file that wraps `docker run`.

## Options Considered

### Option A: Docker Compose (chosen)

- `restart: unless-stopped` in `compose.yaml` delegates autostart to the Docker daemon,
  which itself starts on boot via systemd
- Lifecycle managed entirely through `just compose-up/down/logs/restart`
- compose.yaml is version-controlled in the repo; no `/etc/systemd/` files to manage
- Portable across machines that have Docker Compose V2 installed

### Option B: systemd unit (`/etc/systemd/system/triton-trtllm.service`)

- More "Linux-native" and visible to `systemctl status`
- No Docker Compose dependency; works with plain `docker run`
- Requires writing and enabling a systemd unit file (outside the repo)
- Harder to update atomically when run flags change

## Decision

**Option A — Docker Compose** was chosen because:

1. The lifecycle interface (`just compose-*`) is already defined and version-controlled
2. No files outside the repo need to be managed on the host
3. The Docker daemon starts on boot via systemd already; piggybacking on that is simpler
4. Compose V2 is already required for the project

## Rejected Alternative Preserved for Reference

The equivalent systemd unit (for ansible-k or future hardening) is:

```ini
[Unit]
Description=Triton TRT-LLM Container
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Restart=always
RestartSec=5
EnvironmentFile=$TRTLLM_HOME/.env

ExecStartPre=-/usr/bin/docker rm -f triton-trtllm
ExecStart=/usr/bin/docker run \
  --name triton-trtllm \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=2g \
  -e HF_TOKEN=${HF_TOKEN} \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $TRTLLM_HOME:/workspace/trtllm \
  -v $TRTLLM_HOME/model_repo:/model_repo \
  -w /workspace \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  bash -lc "sleep infinity"

ExecStop=/usr/bin/docker stop triton-trtllm

[Install]
WantedBy=multi-user.target
```

Enable with:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now triton-trtllm
```

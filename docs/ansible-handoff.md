# Ansible Handoff Log

**Project**: TRT-LLM POC  
**Purpose**: Running log of every host-level change made during this POC, in a form
suitable for promotion to `/home/ken/src/config-src/ansible-k`.

> **Constitution II**: Any modification to the host machine MUST be captured here.
> Format: tool name, install command, version pinned, purpose.
> Add entries as you go — do not batch at the end.

---

## Tools Installed

| Tool | Install Command | Version | Purpose |
|------|----------------|---------|---------|
| nvidia-container-toolkit | `sudo apt-get install -y nvidia-container-toolkit` (after adding libnvidia-container repo — see §0.2 setup.md) | 1.19.0-1 | Exposes GPU to Docker containers via `--gpus all`; required before `just dev` |
| libnvidia-container1 | (pulled automatically with nvidia-container-toolkit) | 1.19.0-1 | Core userspace library for GPU container isolation |
| nvidia-container-toolkit-base | (pulled automatically with nvidia-container-toolkit) | 1.19.0-1 | Base config and CDI generation (`nvidia-ctk cdi generate`) |
| libnvidia-container-tools | (pulled automatically with nvidia-container-toolkit) | 1.19.0-1 | CLI tools including `nvidia-ctk` |
| Docker Compose V2 | Pre-installed with Docker Engine (plugin) | v5.1.3 | `docker compose` used by all `just compose-*` recipes |
| `just` task runner | Pre-installed (verify: `just --version`) | 1.50.0 | Repo task runner for all lifecycle commands |
| Rust stable toolchain | Pre-installed via rustup (verify: `cargo --version`) | 1.95.0 (rustc 1.95.0, cargo 1.95.0) | Required for rust-client build |

---

## System Configuration Changes

| Change | File / Service | Content / Diff | Dependency |
|--------|---------------|----------------|------------|
| Added libnvidia-container apt repo | `/etc/apt/sources.list.d/nvidia-container-toolkit.list` | Signed entry for `https://nvidia.github.io/libnvidia-container/stable/deb/` | Must precede toolkit install |
| Added nvidia runtime to Docker | `/etc/docker/daemon.json` | `sudo nvidia-ctk runtime configure --runtime=docker` | nvidia-container-toolkit installed |
| Removed flaky neovim PPA | `/etc/apt/sources.list.d/ppa_neovim_ppa_unstable_noble.list` | PPA removed; `sudo add-apt-repository --remove ppa:neovim-ppa/unstable` + `sudo rm` of list file | `neovim-ppa/unstable` was timing out during `apt-get update`, blocking all other installs |
| *(record additional changes as made)* | | | |

---

## Docker / Container Runtime

| Item | Details |
|------|---------|
| Track A image | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` |
| Track A digest | `sha256:2f0b65e1e33bbe30b352897bf2a1fb9e46120b869809fa5d2991d66c7b259fbe` |
| Track A size | 62.7 GB |
| Track B image | `nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3` |
| Track B digest | `sha256:68a74a08d0839f033befda55b0d9b88da06d943aa3d2bafdb759213bf4427358` |
| Track B size | 52.6 GB |
| NVIDIA Container Toolkit | 1.19.0-1 (via libnvidia-container apt repo) |
| Docker Compose V2 | v5.1.3 (Docker Engine plugin) |
| Docker daemon config | `/etc/docker/daemon.json` — written by `sudo nvidia-ctk runtime configure --runtime=docker` |

---

## Directory Layout on Host

| Path | Purpose | Created By |
|------|---------|-----------|
| `/ai/trtllm-poc/` | TRTLLM_HOME root | `just init` |
| `/ai/trtllm-poc/models/` | Downloaded HuggingFace weights | `just init` |
| `/ai/trtllm-poc/engines/` | Built TRT-LLM engine files | `just init` |
| `/ai/trtllm-poc/cache/` | HuggingFace model cache | `just init` |
| `/ai/trtllm-poc/logs/` | Container stdout logs | `just init` |
| `/ai/trtllm-poc/model_repo/` | Triton model repository | manual / T023 template |

---

## Notes

### neovim-ppa/unstable — flaky PPA, ansible-k defensive handling needed

During NVIDIA Container Toolkit install, `apt-get update` hung/timed out because
`ppa:neovim-ppa/unstable` was unreachable. Unblocked by:

```bash
sudo add-apt-repository --remove ppa:neovim-ppa/unstable
sudo rm /etc/apt/sources.list.d/ppa_neovim_ppa_unstable_noble.list
```

**Ansible-k implication**: The next `ansible-k` deploy will likely re-add this PPA.
Consider adding defensive handling in the role/task that manages this PPA:
- Use `retries:` + `delay:` on the `apt` or `apt_repository` task
- Or gate the PPA on a variable/tag so it can be skipped when the PPA is known-flaky
- Or switch to a different neovim source (e.g., the official AppImage or GitHub release)

### nvidia-cdi-refresh.service failure during toolkit install — kernel upgrade in progress

`nvidia-container-toolkit 1.19.0-1` install triggered a kernel upgrade from
`6.8.0-110-generic` → `6.8.0-111-generic`. The install upgraded the NVIDIA userspace
libraries (`libnvidia-ml.so`) to match the new kernel, but the running module was still
the old version. `nvidia-ctk cdi generate` failed with NVML error `result=9`
(`NVML_ERROR_DRIVER_NOT_LOADED`) — userspace/kernel driver versions mismatched.

**Resolution**: reboot to load `6.8.0-111-generic` + matching NVIDIA module, then:

```bash
# Verify CDI service auto-ran successfully on boot:
systemctl status nvidia-cdi-refresh.service

# Register nvidia runtime with Docker (still required even with CDI):
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Confirm:
docker info | grep -i "runtime\|nvidia"
```

**Ansible-k implication**: The `nvidia-ctk runtime configure` step must run *after* any
kernel upgrade + reboot that touches NVIDIA packages. Order dependency:
`install toolkit → reboot if kernel upgraded → configure runtime → restart docker`.

### Track A inference smoke test — VRAM observations (Qwen2.5-Coder-7B, float16)

Observed during first inference run (SC-002 validation):

| Item | Value |
|------|-------|
| Engine size on GPU | 14,550 MiB (~14.2 GB) |
| KV cache allocated | ~14.0 GB (262,528 tokens, 32 tokens/block) |
| Total GPU memory | 31.35 GiB |
| Available after engine + KV cache | ~1.3 GB headroom |
| Engine load time | 3.74 seconds |

With all three models loaded concurrently (qwen-coder + llama3-chat + llava-caption),
VRAM will be tight. Load one at a time during initial testing; document whether
concurrent serving fits within 32 GB.

### `run.py` / `trtllm-build` workflow marked legacy in 1.3.0rc13

The `run.py` inference script now emits a `FutureWarning`:

> "This is part of the legacy TensorRT engine-build workflow. New projects should use
> the PyTorch backend instead: `trtllm-serve <model_name_or_path>`"

**Ansible-k / future implication**: The `convert_checkpoint + trtllm-build + run.py`
pipeline (Track A) may be deprecated in a future release in favour of `trtllm-serve`
which handles conversion and serving internally. Track B (Triton) is still the correct
production serving path and is unaffected. If Track B proves complex, `trtllm-serve`
is a viable fallback for POC smoke testing.

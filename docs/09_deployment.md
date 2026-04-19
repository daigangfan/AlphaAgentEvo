# 09 · Deployment and Environments

This doc covers setting up the **runtime environment** for each of
the two training tracks and the known gotchas that have been resolved
(and may recur).

## Local machine — Windows + RTX 5090

The current development machine is Windows 11 with PowerShell 7 at
`F:\projects\AlphaAgentEvo`.

### Python and package environment

- Python ≥ 3.12 required (see `pyproject.toml::requires-python`).
- Recommended: [`uv`](https://github.com/astral-sh/uv) managed venv
  (the project ships `.venv/` + `uv.lock`).
- PyTorch is pinned to the CUDA 12.4 wheels via a `[tool.uv.sources]`
  index in `pyproject.toml`.

```powershell path=null start=null
# Install deps (first time)
uv sync

# Or manually
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Running the stack (local `trl` track)

Two-terminal workflow:

```bash path=null start=null
# Terminal 1 — API (blocks)
bash start_api.sh
# Alternative (direct uvicorn):
python -m uvicorn backtest.api_server:app --host 0.0.0.0 --port 8001

# Terminal 2 — training (blocks)
bash start_training.sh
```

`start_training.sh` expects `bash`. On Windows you have three options:

1. Use **Git Bash** or **WSL** to run the `.sh` directly.
2. Replicate the two commands manually in PowerShell:
   ```powershell path=null start=null
   python -m uvicorn backtest.api_server:app --host 0.0.0.0 --port 8001
   # in another tab:
   python training/train.py --config configs/grpo_config.yaml
   ```
3. Use Warp's workflows (see [Warp docs](https://docs.warp.dev)).

### Environment check

```bash path=null start=null
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import pandas as pd; print(pd.read_hdf(r'backtest/data/daily_pv.h5').shape)"
curl -s http://localhost:8001/health
```

## Cloud — Runpod + 4× H100 (Verl track)

### Target pod

- GPUs: 4× NVIDIA H100 80 GB HBM3 (minimum; paper uses 10× RTX 4090 or
  8× A100 for `batch_size=20`).
- Network volume mounted at `/workspace` so state survives pod
  restarts.
- SSH: `ssh <pod>@ssh.runpod.io -i ~/.ssh/id_ed25519`.

### One-time setup

```bash path=null start=null
# Clone into the persistent volume
git clone https://github.com/hongha5192-bit/AlphaAgentEvo.git /workspace/AlphaAgentEvo
cd /workspace/AlphaAgentEvo

# Upload daily_pv.h5 once (from local machine)
scp backtest/data/daily_pv.h5 <pod>:/workspace/AlphaAgentEvo/backtest/data/

# Run the v2 setup (creates conda env verl041, installs pinned Verl v0.4.1)
bash deploy/v2/setup.sh
```

What `deploy/v2/setup.sh` does, numbered to match the script's log
steps:

1. Creates conda env `verl041` with Python 3.10.
2. Clones `verl-project/verl@v0.4.1`.
3. Installs `torch==2.6.0`, `sglang==0.4.6.post5`, `tensordict<=0.6.2`,
   and Verl itself (`pip install --no-deps -e .`).
4. Patches `flash_attn.bert_padding` imports out of the actor / critic
   code (flash-attn has ABI issues with torch 2.6.0).
5. Installs `fastapi`, `uvicorn`, `pyparsing`, `joblib`, and other
   Python deps; and `libnuma-dev` via `apt`.
6. Copies `factor_tool.py`, `factor_tool_config.yaml`, and
   `factor_reward.py` into the Verl source tree.
7. Verifies imports and GPU availability (`python -c "import verl; ..."`).
8. Freezes the env to `$WORK/requirements.lock.txt`.

After setup, `/workspace/v2/verl.tag` contains `v0.4.1`.

### Running training

```bash path=null start=null
# Start API (backgrounded, port 8002)
nohup python /workspace/AlphaAgentEvo/deploy/api_server_verl.py \
    > /workspace/v2/logs/api.log 2>&1 &

# Start training
nohup bash /workspace/AlphaAgentEvo/deploy/v2/train.sh \
    > /workspace/v2/logs/train.log 2>&1 &

# Monitor
tail -f /workspace/v2/logs/train.log
```

### First-run smoke test

Before committing to 150 steps, verify:

```bash path=null start=null
# API works
curl -s http://localhost:8002/health | jq
curl -s -X POST http://localhost:8002/backtest \
    -H "Content-Type: application/json" \
    -d '{"exprs":{"mom":"RANK(TS_PCTCHANGE($close, 20))"}}' | jq '.data.metrics.Information_Ratio_with_cost'

# Tool + reward smoke
python -c "from deploy.v2.factor_tool import *; print('ok')"
```

### Known deployment traps (and resolutions)

From `STATUS.md` "Phase 2" issues (every one has bitten us at some
point):

| # | Problem | Resolution |
|---|---|---|
| 1 | `libnuma.so.1` not found | `apt-get install -y libnuma-dev` at the top of `train.sh`. |
| 2 | `flash_attn` undefined symbol | Skip install; Verl code is patched to fall back to SDPA. |
| 3 | Port mismatch (8001 vs 8002) | Verl API fixed to 8002; don't change it. |
| 4 | `factor_reward.py` not copied | Added explicit `install -D` in `setup.sh`. |
| 5 | `tool_config/` dir missing | `install -D` creates the parent dir. |
| 6 | `daily_pv.h5` is a broken symlink | Upload the real file once over scp. |
| 7 | API not running after restart | Training script re-health-checks and auto-starts if missing (`run_training.sh`). |
| 8 | `ContinueGenerationReqInput` missing | Need sglang 0.4.6.post5, not latest; pin via `v0.4.1/setup.py`. |
| 9 | Verl install script inconsistent | Use `pip install -e .`, not `scripts/install.sh`. |
| 10 | Disk quota exceeded | `conda env remove -n verl041` first, then reinstall. |
| 11 | Wrong Python (3.11 vs 3.10) | Ensure `conda activate verl041` **before** pip commands. |
| 12 | flash-attn wheel wrong cpXYZ | Don't install flash-attn; see #2. |
| 13 | Conda TOS not accepted | `conda tos accept --override-channels --channel main` + `-r`. |

### Lessons learned

From `STATUS.md`:

> For production ML training on cloud GPUs, use the project's official
> Docker image — not a hand-built env.
>
> Verl provides `verlai/verl:sgl055.latest` with everything pre-tested.
> We spent hours debugging dependency mismatches that wouldn't exist
> with the Docker image.
>
> If you must build from source, **always pin to a release tag** and
> install from `setup.py`, not the helper scripts.

## Secrets / credentials

- No cloud credentials or API keys are required for training — the
  backtest is local (runs on the pod) and the model is downloaded
  from Hugging Face (`Qwen/Qwen3-4B-Thinking-2507`).
- If you're pulling the model from a private mirror, set
  `HF_HOME=/workspace/hf-cache` and `HF_TOKEN=$HF_TOKEN` **only** via
  an env var, never in committed files.

## Monitoring & debugging

| What | Where |
|---|---|
| Training log | `/workspace/v2/logs/train.log` (Verl) or stdout (local) |
| API log | `/workspace/v2/logs/api.log` / `logs/api.log` |
| Ray dashboard | `http://<pod>:8265` (Verl uses Ray) |
| GPU usage | `watch -n 2 nvidia-smi` |
| TensorBoard | `tensorboard --logdir outputs/ --bind_all` (local) or `tensorboard --logdir $WORK/checkpoints/ --bind_all` (Verl) |

## Clean restart

```bash path=null start=null
# Kill stale Ray / training processes
ray stop --force
pkill -9 -f 'verl|sglang|ray|raylet' || true
rm -rf /tmp/ray /dev/shm/ray*

# Kill stale API
pkill -f api_server || true

# Start fresh
nohup python /workspace/AlphaAgentEvo/deploy/api_server_verl.py > /workspace/v2/logs/api.log 2>&1 &
nohup bash /workspace/AlphaAgentEvo/deploy/v2/train.sh          > /workspace/v2/logs/train.log 2>&1 &
```

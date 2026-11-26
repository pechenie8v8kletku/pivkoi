# Local real-model test setup

Completed full-run results: `real-model-results.md` (580 evaluated responses,
all ten technical checks passed; quality limitations reported explicitly).

The optional GPU runtime lives in ignored `local_experiments/qwen-runtime`.
It is separate from the service virtual environment. Confirmed locally:
RTX 5070, CUDA PyTorch 2.9.1+cu128, BF16 capability and CUDA matrix multiplication.
Model inference and quality results must be recorded separately after weights load.

## Verified initial inference

Both models loaded together on the RTX 5070 without quantization. Initial
`scripts/model_smoke.py` passed all seven checks: real 2560-dimensional finite
unit embeddings, batch/single consistency, relevant-over-unrelated ordering
for both Qwen and BGE, and no CUDA OOM. This small smoke test does not establish
catalog recommendation quality; the full TCP audit runs separately.

- Allocated GPU memory: 8.558 GiB; peak allocated: 8.939 GiB.
- Reserved allocator memory: 9.031 GiB; batch size 2, max observed input 292 tokens.
- Query about bread dough ingredients: cosine with flour 0.556 versus pumps 0.160.
- BGE logits for the same pair: -2.623 versus -11.031 (not probabilities).
- Batch-versus-single cosine: 0.999865 / 0.999820.
- Source report: ignored `local_experiments/model-smoke.json`.

Pinned downloads: Qwen `5cf2132abc99cad020ac570b19d031efec650f2b`;
BGE `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

Model artifacts are pinned in `local_experiments/models/manifest.json` by
the downloader. Do not commit model weights, runtime environments or synthetic databases.

## Start after download completion

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
$env:HF_HOME = Join-Path $PWD 'local_experiments/hf-cache'
local_experiments/qwen-runtime/Scripts/python.exe scripts/local_model_server.py --reranker --embedding-path local_experiments/models/qwen3-embedding-4b --reranker-path local_experiments/models/bge-reranker-v2-m3 --fixtures local_experiments/20261126-123951-quality/extractions.json
```

Server binds only to 127.0.0.1:8011. Qwen runs BF16, BGE FP16, with a shared
GPU inference lock and batch size 2. Both models must actually load and execute;
there is no silent CPU or mock embedding fallback. The 512-token service cap
is a local memory budget, not the model's maximum context. Oversize inputs are
rejected instead of silently truncated. Health reports inference counters and VRAM.

```powershell
.venv/Scripts/python.exe scripts/model_smoke.py
.venv/Scripts/python.exe scripts/real_model_audit.py --fixtures local_experiments/20261126-123951-quality
```

The audit starts a real single-worker service process on loopback port 8012,
persists SQLite and restarts the process for each mode. It compares signals
off/on, BGE off/on and semantic/online rankers, then sends impressions, dwell,
opens, likes, contacts, hides and duplicate event retries. Fixed decision-time
features measure personal score changes; a second user checks isolation.
The shared segment graph may legitimately change another user's live feed.
Concurrent requests, hidden targets and trained state are checked across restart.

This is a production-path local integration experiment, NOT full production:
chat/profile extraction is explicitly scripted, events are synthetic, SQL is
SQLite rather than PostgreSQL, one worker runs, and there is no WAN/load-cluster
simulation. Qwen/BGE weights are not fine-tuned; only the service's personal
ranking model learns from events. The latency budget overrides remove API
rate-limit waits for this local benchmark; they are not deployment defaults.

Reference inference implementations: official model cards
`Qwen/Qwen3-Embedding-4B` (left padding, last-token pooling, normalization) and
`BAAI/bge-reranker-v2-m3` (sequence-classification logits).

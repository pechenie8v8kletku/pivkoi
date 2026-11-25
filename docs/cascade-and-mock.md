# Recommendation cascade: MVP and evolution path

## Component layout

```text
Client snapshot
  └─ profile builder (LLM once per semantic revision)
      └─ multivector index (need / offer / product)

Viewer context
  └─ candidate generators (directed semantic matching; later graph/CF/fresh pools)
      └─ feature builder (decision-time snapshot)
          └─ BGE pair reranker (optional provider stage)
              └─ ranker strategy (semantic | linucb | future global model)
                  └─ slate policy (MMR + novelty-weighted exploration)
                      └─ feed/search response + immutable slate log
                          └─ event collector → online state + raw event log
```

The boundaries are in `indexing.py`, `engine.py`, `ranking.py`, `relations.py`,
`providers.py`, and `storage.py`. Stage interfaces are intentionally kept replaceable;
this is a YouTube-style retrieval/ranking architecture, not a reproduction of YouTube's
models or training infrastructure.

## MVP execution path

1. **Cold-start retrieval:** separate needs/offers/products; directed candidate matching.
2. **BGE:** score exact query/need ↔ offer/product pairs for up to `RERANK_TOP_K`
   candidates after retrieval. Feed and search can use the same stage. Failure falls back.
3. **Feature vector:** pair scores, graph edge evidence, freshness, geography, stock,
   catalog size, prior exposure, novelty, and intent. Persist the exact vector with each slate.
4. **Ranker:** select `RANKER_MODE=semantic` for deterministic semantic baseline,
   or `linucb` for per-viewer regularized linear online personalization.
5. **Slate policy:** MMR-like redundancy penalty plus a configurable epsilon-greedy
   branch weighted toward fresh/less-seen candidates. Logs the conditional probability
   of each choice at its selection step.
6. **Feedback:** logged `open`, `like`, `contact`, `hide`, sufficiently long `dwell`
   update the client's ranker state. Impression is logged and may update the aggregate
   segment graph, but is not a personal negative label. No action is not negative.

The linear ranker is a small individual model, not a global trained ranking model.
It starts from the semantic baseline and updates residual reward with ridge regression.
It does not change Qwen/BGE weights. All initial reward and policy settings are visible
in config and are uncalibrated starting values. Do not report them as learned business
metrics until validated on real impressions and outcomes.

## Runtime switches

```dotenv
RANKER_MODE=linucb            # semantic | linucb
MODEL_API_MODE=mock           # mock | external
RERANK_ENABLED=true
RERANK_FEED_ENABLED=true
RERANK_TOP_K=12
LINUCB_REGULARIZATION=1.0
LINUCB_ALPHA=0.15
RANKER_PERSONAL_WEIGHT=0.5
RANKER_RERANK_WEIGHT=0.35
RANKER_GRAPH_WEIGHT=0.1
RANKER_FRESHNESS_WEIGHT=0.03
RANKER_REGION_WEIGHT=0.02
DIVERSITY_WEIGHT=0.12
EXPLORATION_RATE=0.1
EXPLORATION_FRESHNESS_WEIGHT=1.0
FRESHNESS_HALF_LIFE_DAYS=14
```

`semantic` disables personal linear updates but retains the feature/candidate/slate
contracts. `linucb` enables incremental user-specific feedback. `MODEL_API_MODE` is
independent of `RANKER_MODE`: local mock or real API models can be paired with either.

## Run mock model API

Terminal 1:
```powershell
.venv\Scripts\python -m business_match.mock_api
```

Terminal 2:
```powershell
.venv\Scripts\python -m uvicorn business_match.app:app --port 8000 --workers 1
```

Default `.env.example` points to the mock. It implements:

| Route | Purpose | Contract |
| --- | --- | --- |
| `GET /health` | Mock health | deterministic-mock |
| `POST /v1/embeddings` | OpenAI-compatible embed | `model`, `input`; returns indexed vectors |
| `POST /v1/chat/completions` | OpenAI-compatible structured output | tool call with profile JSON |
| `POST /v1/rerank` | Pairwise cross-encoder-shaped API | `pairs[{query,document}]`; indexed scores |

The embedding mock uses deterministic signed feature hashing, the reranker cosine over
those mock vectors, and the LLM returns a predictable profile from the input. These
stubs verify API plumbing, persistence, and cascade flow only; they do not approximate
Qwen/BGE semantic quality and must not be used to tune relevance or policy parameters.

For real models set `MODEL_API_MODE=external`, configure each provider URL and key,
and confirm the real reranker gateway supports the pair contract. Model traffic goes
through the same replaceable provider adapter. Qwen model names remain those supplied
for the hackathon.

## Future global ranking

Keep the feature schema and raw decision/event logs as training data. When enough
exposures accrue:

1. Validate impression visibility and action attribution; add propensity-aware randomized
   exploration if causal/offline evaluation is required.
2. Train and validate a shared `LightGBM LambdaMART` or CatBoost ranking model using
   pointwise outcomes or grouped rankings. Keep per-user state as a small residual or
   context overlay initially.
3. Add candidate sources independently: taxonomy/segment graph, collaborative neighbors,
   and fresh/new businesses. Deduplicate/merge candidates before feature building.
4. Add model versioning, offline metrics and shadow/canary deployment behind a new
   `RANKER_MODE` value; keep `semantic` and `linucb` as rollback paths.
5. Consider learned/two-tower retrieval only once actual interactions are plentiful.

The sparse directed segment graph already accumulates observed evidence, but its schema
and weights are an initial aggregate, not a trained embedding or causal graph. Replace
free-text segment keys with stable taxonomy IDs before treating edges as production
features.

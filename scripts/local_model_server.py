"""Local single-GPU Qwen/BGE HTTP gateway; optional explicit scripted chat fixtures.

Run in the isolated local_experiments/qwen-runtime environment. Never pretends
that scripted profiles are a real Qwen chat model. Model weights stay in HF_HOME.
"""

import argparse
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

EMBEDDING = "Qwen/Qwen3-Embedding-4B"
RERANKER = "BAAI/bge-reranker-v2-m3"


class EmbeddingRequest(BaseModel):
    model: str = EMBEDDING
    input: str | list[str]


class RerankRequest(BaseModel):
    model: str = RERANKER
    pairs: list[dict[str, str]] = Field(min_length=1, max_length=64)


def build_app(args):
    state = {
        "ready": False,
        "embedding_calls": 0,
        "rerank_calls": 0,
        "embedding_texts": 0,
        "rerank_pairs": 0,
        "ooms": 0,
        "max_embedding_tokens": 0,
        "max_rerank_tokens": 0,
    }
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; no silent CPU/mock fallback")
        state["tokenizer"] = AutoTokenizer.from_pretrained(args.embedding_path, padding_side="left")
        state["embedding"] = AutoModel.from_pretrained(
            args.embedding_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        ).eval()
        if args.reranker:
            state["rerank_tokenizer"] = AutoTokenizer.from_pretrained(args.reranker_path)
            state["reranker"] = AutoModelForSequenceClassification.from_pretrained(
                args.reranker_path, torch_dtype=torch.float16, device_map={"": "cuda:0"}
            ).eval()
        state["ready"] = True
        print(
            json.dumps(
                {
                    "ready": True,
                    "gpu": torch.cuda.get_device_name(0),
                    "allocated_gb": torch.cuda.memory_allocated() / 2**30,
                }
            ),
            flush=True,
        )
        yield

    app = FastAPI(title="Local real Qwen embeddings and BGE reranker", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            **{k: v for k, v in state.items() if isinstance(v, (str, int, float, bool))},
            "embedding_model": EMBEDDING,
            "reranker_model": RERANKER if args.reranker else None,
            "embedding_revision": getattr(state.get("embedding"), "config", None)._commit_hash
            if state.get("embedding") is not None
            else None,
            "gpu": torch.cuda.get_device_name(0),
            "allocated_gb": torch.cuda.memory_allocated() / 2**30,
            "reserved_gb": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "chat_mode": "scripted-fixture" if args.fixtures else "unavailable",
        }

    @app.post("/v1/embeddings")
    def embed(request: EmbeddingRequest):
        if request.model not in (EMBEDDING, "Qwen3-Embedding-4B"):
            raise HTTPException(422, "Unknown embedding model")
        texts = [request.input] if isinstance(request.input, str) else request.input
        if not texts or len(texts) > 128 or any(not t or len(t) > 32000 for t in texts):
            raise HTTPException(422, "Invalid texts or request budget")
        vectors, tokens = [], 0
        with lock, torch.inference_mode():
            try:
                for start in range(0, len(texts), args.batch_size):
                    encoded = state["tokenizer"](
                        texts[start : start + args.batch_size],
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                    length = encoded["input_ids"].shape[1]
                    if length > args.max_length:
                        raise HTTPException(422, "Token limit exceeded; no silent truncation")
                    state["max_embedding_tokens"] = max(state["max_embedding_tokens"], length)
                    tokens += int(encoded["attention_mask"].sum())
                    encoded = encoded.to("cuda:0")
                    output = state["embedding"](**encoded, use_cache=False)
                    # Left padding: final token is the final real token of every sequence.
                    pooled = F.normalize(output.last_hidden_state[:, -1].float(), p=2, dim=1)
                    vectors.extend(pooled.cpu().tolist())
                    del output, encoded, pooled
                state["embedding_calls"] += 1
                state["embedding_texts"] += len(texts)
            except torch.cuda.OutOfMemoryError as exc:
                state["ooms"] += 1
                torch.cuda.empty_cache()
                raise HTTPException(
                    503, "CUDA out of memory; reduce configured batch size"
                ) from exc
        return {
            "object": "list",
            "model": EMBEDDING,
            "data": [
                {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    @app.post("/rerank")
    @app.post("/v1/rerank")
    def rerank(request: RerankRequest):
        if not args.reranker:
            raise HTTPException(503, "BGE not loaded")
        if request.model not in (RERANKER, "bge-reranker-v2-m3"):
            raise HTTPException(422, "Unknown reranker model")
        if any(not p.get("query") or not p.get("document") for p in request.pairs):
            raise HTTPException(422, "Each pair needs query and document")
        values = []
        with lock, torch.inference_mode():
            try:
                for start in range(0, len(request.pairs), args.batch_size):
                    pairs = [
                        [p["query"], p["document"]]
                        for p in request.pairs[start : start + args.batch_size]
                    ]
                    encoded = state["rerank_tokenizer"](
                        pairs, padding=True, truncation=False, return_tensors="pt"
                    )
                    length = encoded["input_ids"].shape[1]
                    if length > args.max_length:
                        raise HTTPException(422, "Reranker token limit exceeded")
                    state["max_rerank_tokens"] = max(state["max_rerank_tokens"], length)
                    encoded = encoded.to("cuda:0")
                    scores = state["reranker"](**encoded).logits.view(-1).float()
                    values.extend(scores.cpu().tolist())
                    del encoded, scores
                state["rerank_calls"] += 1
                state["rerank_pairs"] += len(request.pairs)
            except torch.cuda.OutOfMemoryError as exc:
                state["ooms"] += 1
                torch.cuda.empty_cache()
                raise HTTPException(503, "CUDA out of memory in BGE") from exc
        return {
            "model": RERANKER,
            "results": [{"index": i, "relevance_score": v} for i, v in enumerate(values)],
        }

    @app.post("/v1/chat/completions")
    def scripted_chat(body: dict):
        if not args.fixtures:
            raise HTTPException(503, "No chat model deployed")
        # Deliberately separate deterministic extraction from genuine neural inference.
        from business_match.mock_api import chat_completions

        result = chat_completions(body)
        tool = result["choices"][0]["message"]["tool_calls"][0]["function"]
        if "commercial" in json.loads(tool["arguments"]):
            source = json.loads(
                next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
            )
            answers = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
            value = answers.get(
                source["subject"],
                {"subject": "", "role": "unknown", "commercial": False, "confidence": 0},
            )
            tool["arguments"] = json.dumps(value, ensure_ascii=False)
        return result

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--embedding-path", default=EMBEDDING)
    parser.add_argument("--reranker-path", default=RERANKER)
    parser.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    uvicorn.run(build_app(args), host="127.0.0.1", port=args.port, workers=1)

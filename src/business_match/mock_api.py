"""Local mock for the configured Qwen embeddings/chat and BGE reranker HTTP APIs."""

import hashlib
import json
import re
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DIMENSIONS = 384


def token_features(text: str) -> list[str]:
    normalized = text.casefold().replace("ё", "е")
    words = re.findall(r"[\w-]+", normalized, flags=re.UNICODE)
    features = [f"w:{word}" for word in words]
    for word in words:
        padded = f"^{word}$"
        features.extend(f"c:{padded[i:i + 3]}" for i in range(max(1, len(padded) - 2)))
    return features or ["empty"]


def mock_embedding(text: str) -> list[float]:
    vector = np.zeros(DIMENSIONS, dtype=np.float64)
    for feature in token_features(text):
        digest = hashlib.blake2b(feature.encode(), digest_size=16).digest()
        index = int.from_bytes(digest[:4], "little") % DIMENSIONS
        sign = 1 if digest[4] & 1 else -1
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector.tolist()


def client_profile(source: dict[str, Any]) -> dict[str, Any]:
    name = source.get("name", "Компания")
    segment = source.get("segment", "бизнеса")
    product_texts = []
    for product in source.get("products", [])[:40]:
        if product.get("active", True) and product.get("stock") != 0:
            product_texts.append(" ".join(filter(None, [
                product.get("name"), product.get("description"), product.get("category")
            ])))
    fallback_offer = source.get("description") or f"Предложение компании {name}: {segment}"
    offers = product_texts or [fallback_offer]
    needs = [{"text": text, "intent": "supplier", "importance": 3, "evidence": "explicit"}
             for text in source.get("explicit_needs", [])[:15]]
    if not needs:
        needs = [
            {"text": f"Поставщики товаров и услуг для бизнеса: {segment}",
             "intent": "supplier", "importance": 2, "evidence": "inferred"},
            {"text": f"Покупатели продукции компании {name}: {segment}",
             "intent": "buyer", "importance": 2, "evidence": "inferred"},
            {"text": f"Партнёры для бизнеса {segment}",
             "intent": "partner", "importance": 1, "evidence": "inferred"},
        ]
    return {
        "summary": source.get("description") or f"{name}: компания сегмента {segment}.",
        "offers": offers[:15],
        "needs": needs,
    }


def response_for_schema(schema: dict, source: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    if "commercial" in properties and "confidence" in properties:
        # No invented semantic capability: tests inject controlled extractor responses.
        return {"subject": "", "role": "unknown", "commercial": False, "confidence": 0}
    profile = client_profile(source)
    output = {}
    for key, definition in properties.items():
        if key in profile:
            output[key] = profile[key]
        elif definition.get("type") == "array":
            output[key] = []
        elif definition.get("type") == "string":
            output[key] = "Mock response"
        elif definition.get("type") in ("integer", "number"):
            output[key] = 1
        elif definition.get("type") == "boolean":
            output[key] = False
        else:
            output[key] = None
    return output


app = FastAPI(title="Business Match Model API Mock", version="0.1.0")


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


class RerankRequest(BaseModel):
    model: str
    pairs: list[dict[str, str]]


@app.get("/health")
def health():
    return {"status": "ok", "mode": "deterministic-mock"}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts or any(not isinstance(text, str) for text in texts):
        raise HTTPException(422, "input must contain text")
    return {"object": "list", "model": request.model, "data": [
        {"object": "embedding", "index": i, "embedding": mock_embedding(text)}
        for i, text in enumerate(texts)
    ], "usage": {"prompt_tokens": sum(len(text.split()) for text in texts), "total_tokens": 0}}


@app.post("/rerank")
@app.post("/v1/rerank")
def rerank(request: RerankRequest):
    results = []
    for index, pair in enumerate(request.pairs):
        query = mock_embedding(pair["query"])
        document = mock_embedding(pair["document"])
        results.append({"index": index, "relevance_score": float(np.dot(query, document))})
    return {"model": request.model, "results": results}


@app.post("/v1/chat/completions")
def chat_completions(payload: dict):
    messages = payload.get("messages", [])
    tools = payload.get("tools", [])
    if not tools:
        raise HTTPException(400, "Mock chat endpoint supports structured tool output only")
    user_content = next((m.get("content", "") for m in reversed(messages)
                         if m.get("role") == "user"), "")
    try:
        source = json.loads(user_content)
    except (TypeError, json.JSONDecodeError):
        source = {}
    function = tools[0].get("function", {})
    name = function.get("name", "final_result")
    arguments = response_for_schema(function.get("parameters", {}), source)
    return {
        "id": "chatcmpl-local-mock",
        "object": "chat.completion",
        "created": 0,
        "model": payload.get("model", "local-mock"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-profile-mock",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }]},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("business_match.mock_api:app", host="127.0.0.1", port=8010, reload=False)

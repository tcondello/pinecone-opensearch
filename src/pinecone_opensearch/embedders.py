"""Embedder registry: Pinecone Inference or callables (M3)."""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]: ...


class CallableEmbedder:
    def __init__(self, fn: Callable[..., Any], *, sparse: bool = False):
        self._fn = fn
        self.sparse = sparse

    def embed(self, texts: list[str], **kwargs: Any) -> list[Any]:
        result = self._fn(texts, **kwargs)
        return list(result)


class PineconeInferenceEmbedder:
    """Thin wrapper around ``pc.inference.embed``."""

    def __init__(self, pc: Any, *, model: str, sparse: bool = False, **defaults: Any):
        self._pc = pc
        self.model = model
        self.sparse = sparse
        self.defaults = defaults

    def embed(self, texts: list[str], **kwargs: Any) -> list[Any]:
        params = {**self.defaults, **kwargs}
        resp = self._pc.inference.embed(model=self.model, inputs=texts, parameters=params or None)
        # SDK returns objects with .values or sparse indices/values
        out = []
        data = getattr(resp, "data", None) or resp
        for item in data:
            if self.sparse:
                out.append({
                    "indices": list(getattr(item, "sparse_indices", None)
                                    or getattr(item, "indices", None)
                                    or item.get("indices", [])),
                    "values": list(getattr(item, "sparse_values", None)
                                   or getattr(item, "values", None)
                                   or item.get("values", [])),
                })
            else:
                values = getattr(item, "values", None)
                if values is None and isinstance(item, dict):
                    values = item.get("values")
                out.append(list(values))
        return out


class EmbedderRegistry:
    def __init__(
        self,
        config_embedders: dict[str, Any] | None = None,
        *,
        pc: Any = None,
        pc_factory: Callable[[], Any] | None = None,
    ):
        self._by_model_id: dict[str, Any] = {}
        self._pc = pc
        self._pc_factory = pc_factory
        for model_id, spec in (config_embedders or {}).items():
            self.register(model_id, spec)

    def register(self, model_id: str, spec: Any) -> None:
        if callable(spec):
            self._by_model_id[model_id] = CallableEmbedder(spec)
            return
        if isinstance(spec, dict):
            kind = spec.get("type") or spec.get("provider") or "pinecone"
            sparse = bool(spec.get("sparse") or kind == "sparse")
            if kind in ("callable", "function") and callable(spec.get("fn")):
                self._by_model_id[model_id] = CallableEmbedder(spec["fn"], sparse=sparse)
                return
            model = spec.get("model") or model_id
            if self._pc is None:
                # Lazy: store spec for later
                self._by_model_id[model_id] = {**spec, "_pending": True, "model": model, "sparse": sparse}
                return
            self._by_model_id[model_id] = _inference_embedder(self._pc, model, sparse, spec)
            return
        # Already an embedder-like object
        self._by_model_id[model_id] = spec

    def get(self, model_id: str) -> Any:
        if model_id not in self._by_model_id:
            raise KeyError(
                f"No embedder configured for model_id={model_id!r}. "
                f"Add it under embedders: in pinecone_opensearch.yaml"
            )
        emb = self._by_model_id[model_id]
        if isinstance(emb, dict) and emb.get("_pending"):
            if self._pc is None and self._pc_factory is not None:
                self._pc = self._pc_factory()
            if self._pc is None:
                raise RuntimeError(
                    f"Embedder {model_id!r} needs a Pinecone client; pass pinecone_api_key"
                )
            self._by_model_id[model_id] = _inference_embedder(
                self._pc, emb.get("model") or model_id, bool(emb.get("sparse")), emb
            )
            emb = self._by_model_id[model_id]
        return emb

    def embed_text(self, model_id: str, text: str) -> tuple[Any, float]:
        """Return ``(vector_or_sparse, embed_ms)``."""
        emb = self.get(model_id)
        started = time.perf_counter()
        vectors = emb.embed([text])
        embed_ms = (time.perf_counter() - started) * 1000.0
        return vectors[0], embed_ms


_SPEC_KEYS = {"type", "provider", "model", "sparse", "fn", "_pending"}


def _inference_embedder(pc: Any, model: str, sparse: bool, spec: dict[str, Any]) -> PineconeInferenceEmbedder:
    params = dict(spec.get("parameters") or {})
    params.update({k: v for k, v in spec.items() if k not in _SPEC_KEYS and k != "parameters"})
    return PineconeInferenceEmbedder(pc, model=model, sparse=sparse, **params)


def get_embedder(registry: EmbedderRegistry, model_id: str) -> Any:
    return registry.get(model_id)

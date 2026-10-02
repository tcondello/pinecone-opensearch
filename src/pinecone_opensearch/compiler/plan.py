"""Plan Pinecone Documents API calls from parsed IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pinecone_opensearch.compiler.filters import FilterCompileError, compile_filters
from pinecone_opensearch.compiler.ir import ParsedBody, ScoringClause
from pinecone_opensearch.compiler.scoring import (
    ScoringCompileError,
    and_match_filter_for_clause,
    compile_score_by,
)
from pinecone_opensearch.config import ClientConfig, IndexConfig
from pinecone_opensearch.exceptions import UnsupportedQueryError
from pinecone_opensearch.mapping import MappingRegistry


@dataclass
class PineconeCall:
    method: str  # "documents.search" | "documents.fetch"
    index: str
    namespace: str
    request: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "index": self.index,
            "namespace": self.namespace,
            "request": self.request,
        }


@dataclass
class TranslationResult:
    plan: str  # fetch | single_search | fan_out
    calls: list[PineconeCall]
    warnings: list[str] = field(default_factory=list)
    pagination: dict[str, int] = field(default_factory=dict)
    fusion: dict[str, Any] | None = None
    logical_index: str = ""
    mode: str = "strict"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "plan": self.plan,
            "calls": [c.to_dict() for c in self.calls],
            "warnings": list(self.warnings),
            "pagination": dict(self.pagination),
            "logical_index": self.logical_index,
            "mode": self.mode,
        }
        if self.fusion is not None:
            out["fusion"] = self.fusion
        return out


def build_plan(
    parsed: ParsedBody,
    *,
    logical_index: str,
    index_cfg: IndexConfig,
    registry: MappingRegistry,
    config: ClientConfig,
    namespace: str | None = None,
) -> TranslationResult:
    warnings = list(parsed.warnings)
    mode = config.mode

    # Untranslatable clauses always raise (all modes). Approximations differ by mode later.
    if parsed.unsupported:
        path, _msg = parsed.unsupported[0]
        detail = "; ".join(f"{p}: {m}" for p, m in parsed.unsupported)
        raise UnsupportedQueryError(path, detail)

    from_ = parsed.from_
    size = parsed.size
    top_k = from_ + size
    max_window = config.max_result_window
    if top_k > max_window:
        raise UnsupportedQueryError(
            "from+size",
            f"from+size ({top_k}) exceeds max_result_window ({max_window})",
        )

    ns = namespace or index_cfg.default_namespace
    pinecone_index = index_cfg.pinecone_index

    try:
        filt, filt_warnings = compile_filters(
            parsed.filter_clauses, index=logical_index, registry=registry
        )
        warnings.extend(filt_warnings)
    except FilterCompileError as exc:
        raise UnsupportedQueryError(exc.path, exc.message) from exc

    # Drop constant_score from "real" scoring for plan selection
    real_scoring = [c for c in parsed.scoring if c.kind != "constant_score"]
    constant = [c for c in parsed.scoring if c.kind == "constant_score"]
    boost = constant[0].boost if constant and not real_scoring else 1.0

    pagination = {"from": from_, "size": size, "top_k": top_k}

    if not real_scoring:
        # fetch plan
        request: dict[str, Any] = {"top_k": top_k}
        if filt is not None:
            request["filter"] = filt
        include = _include_fields(parsed.source, logical_index, registry)
        if include is not None:
            request["include_fields"] = include
        call = PineconeCall(
            method="documents.fetch",
            index=pinecone_index,
            namespace=ns,
            request=request,
        )
        if constant:
            warnings.append(f"Filter-only hits scored with constant boost={boost}")
        else:
            warnings.append("Filter-only / match_all hits scored 1.0")
        return TranslationResult(
            plan="fetch",
            calls=[call],
            warnings=warnings,
            pagination=pagination,
            logical_index=logical_index,
            mode=mode,
        )

    if len(real_scoring) == 1:
        clause = real_scoring[0]
        try:
            score_by, sw = compile_score_by(
                clause, index=logical_index, registry=registry, top_k=top_k
            )
            warnings.extend(sw)
        except ScoringCompileError as exc:
            raise UnsupportedQueryError(exc.path, exc.message) from exc

        extra = and_match_filter_for_clause(clause, index=logical_index, registry=registry)
        merged_filter = _merge_filters(filt, extra)

        request = {
            "score_by": score_by,
            "top_k": _top_k_for_clause(clause, top_k, parsed),
        }
        if merged_filter is not None:
            request["filter"] = merged_filter
        include = _include_fields(parsed.source, logical_index, registry)
        if include is not None:
            request["include_fields"] = include
        if parsed.min_score is not None:
            warnings.append("min_score applied client-side after score conversion")

        call = PineconeCall(
            method="documents.search",
            index=pinecone_index,
            namespace=ns,
            request=request,
        )
        return TranslationResult(
            plan="single_search",
            calls=[call],
            warnings=warnings,
            pagination=pagination,
            logical_index=logical_index,
            mode=mode,
        )

    # N scoring clauses → fan_out
    if mode == "strict" and not _is_hybrid_style(parsed, real_scoring):
        # bool with multiple scoring clauses raises in strict mode
        paths = ", ".join(c.path for c in real_scoring)
        raise UnsupportedQueryError(
            real_scoring[1].path,
            f"Multiple scoring clauses in strict mode ({paths}). Use hybrid or mode: approx",
        )

    calls: list[PineconeCall] = []
    for clause in real_scoring:
        sub_top_k = _top_k_for_clause(clause, top_k, parsed)
        try:
            score_by, sw = compile_score_by(
                clause, index=logical_index, registry=registry, top_k=sub_top_k
            )
            warnings.extend(sw)
        except ScoringCompileError as exc:
            raise UnsupportedQueryError(exc.path, exc.message) from exc

        extra = and_match_filter_for_clause(clause, index=logical_index, registry=registry)
        merged_filter = _merge_filters(filt, extra)
        request = {"score_by": score_by, "top_k": sub_top_k}
        if merged_filter is not None:
            request["filter"] = merged_filter
        include = _include_fields(parsed.source, logical_index, registry)
        if include is not None:
            request["include_fields"] = include
        calls.append(
            PineconeCall(
                method="documents.search",
                index=pinecone_index,
                namespace=ns,
                request=request,
            )
        )

    fusion: dict[str, Any] | None = {
        "strategy": "pipeline",
        "search_pipeline": parsed.search_pipeline,
    }
    if mode == "approx" and not _is_hybrid_style(parsed, real_scoring):
        warnings.append(
            "Multiple bool scoring clauses: approx mode fans out and sums raw scores (approximate)"
        )
        fusion["strategy"] = "sum_raw_scores"
    elif _is_hybrid_style(parsed, real_scoring):
        fusion["strategy"] = "pipeline"

    return TranslationResult(
        plan="fan_out",
        calls=calls,
        warnings=warnings,
        pagination=pagination,
        fusion=fusion,
        logical_index=logical_index,
        mode=mode,
    )


def _top_k_for_clause(clause: ScoringClause, top_k: int, parsed: ParsedBody) -> int:
    if parsed.hybrid_pagination_depth is not None:
        return int(parsed.hybrid_pagination_depth)
    if clause.kind in ("knn", "knn_radial", "neural", "neural_sparse") and clause.params.get("k") is not None:
        return min(int(clause.params["k"]), top_k)
    return top_k


def _is_hybrid_style(parsed: ParsedBody, scoring: list[ScoringClause]) -> bool:
    # Heuristic: hybrid parser puts paths under query.hybrid
    return any(".hybrid." in c.path or c.path.startswith("query.hybrid") for c in scoring)


def _merge_filters(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if a is None:
        return b
    if b is None:
        return a
    return {"$and": [a, b]}


def _include_fields(
    source: Any,
    index: str,
    registry: MappingRegistry,
) -> list[str] | None:
    if source is True or source is None:
        return None
    if source is False:
        return []
    if isinstance(source, list):
        return [registry.rewrite_field(index, f) for f in source]
    if isinstance(source, dict):
        includes = source.get("includes") or source.get("include")
        if includes:
            return [registry.rewrite_field(index, f) for f in includes]
        # excludes applied client-side in M1
        return None
    return None

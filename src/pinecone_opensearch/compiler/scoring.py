"""Compile scoring clauses into Pinecone score_by objects."""

from __future__ import annotations

import re
from typing import Any

from pinecone_opensearch.compiler.ir import ScoringClause
from pinecone_opensearch.mapping import MappingRegistry


class ScoringCompileError(Exception):
    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path
        self.message = message


def compile_score_by(
    clause: ScoringClause,
    *,
    index: str,
    registry: MappingRegistry,
    top_k: int,
) -> tuple[dict[str, Any], list[str]]:
    """Return (score_by, warnings) for a single scoring clause."""
    warnings: list[str] = []
    kind = clause.kind
    params = clause.params
    path = clause.path

    if kind == "knn":
        field = registry.rewrite_field(index, params["field"])
        # Planner supplies effective top_k (min(k, from+size) or hybrid pagination_depth)
        score_by = {
            "dense_vector": {
                "field": field,
                "values": list(params["vector"]),
                "top_k": top_k,
            }
        }
        return score_by, warnings

    if kind == "knn_radial":
        field = registry.rewrite_field(index, params["field"])
        radial_cap = 1000
        score_by = {
            "dense_vector": {
                "field": field,
                "values": list(params["vector"]),
                "top_k": min(radial_cap, top_k) if top_k else radial_cap,
            }
        }
        warnings.append(
            f"{path}: radial knn (min_score/max_distance) applied client-side after score conversion (M1)"
        )
        return score_by, warnings

    if kind == "match":
        field = registry.rewrite_field(index, params["field"])
        operator = params.get("operator") or "or"
        msm = params.get("minimum_should_match")
        if operator == "and" or str(msm) in ("100%", "100"):
            # text score_by + filter $match_all — filter joined by planner
            score_by = {"text": {"fields": [field], "query": params["query"]}}
            warnings.append(f"{path}: operator:and / msm 100% adds $match_all filter on {field}")
            return score_by, warnings
        score_by = {"text": {"fields": [field], "query": params["query"]}}
        return score_by, warnings

    if kind == "match_fuzzy":
        field = registry.rewrite_field(index, params["field"])
        query = str(params["query"])
        fuzz = params.get("fuzziness")
        n = _fuzziness_edits(fuzz, query)
        tokens = query.split()
        qs = " ".join(f"{t}~{n}" for t in tokens)
        score_by = {"query_string": {"query": f"{field}:({qs})"}}
        return score_by, warnings

    if kind == "match_phrase":
        field = registry.rewrite_field(index, params["field"])
        slop = int(params.get("slop") or 0)
        q = str(params["query"]).replace('"', '\\"')
        if slop:
            qs = f'{field}:("{q}"~{slop})'
        else:
            qs = f'{field}:("{q}")'
        score_by = {"query_string": {"query": qs}}
        return score_by, warnings

    if kind == "match_phrase_prefix":
        field = registry.rewrite_field(index, params["field"])
        tokens = str(params["query"]).split()
        if not tokens:
            raise ScoringCompileError(path, "match_phrase_prefix query is empty")
        if len(tokens) == 1:
            qs = f'{field}:("{tokens[0]}"*)'
        else:
            head = " ".join(tokens[:-1])
            qs = f'{field}:("{head} {tokens[-1]}"*)'
        score_by = {"query_string": {"query": qs}}
        return score_by, warnings

    if kind == "multi_match":
        fields = [_parse_boosted_field(f, index, registry) for f in params.get("fields") or []]
        # If any field has a boost, use query_string
        if any(b != 1.0 for _, b in fields) or any("^" in f for f in (params.get("fields") or [])):
            parts = []
            q = str(params["query"]).replace('"', '\\"')
            for fname, boost in fields:
                if boost != 1.0:
                    parts.append(f"({fname}:{q})^{boost:g}")
                else:
                    parts.append(f"{fname}:{q}")
            score_by = {"query_string": {"query": " OR ".join(parts)}}
            return score_by, warnings
        score_by = {
            "text": {
                "fields": [f for f, _ in fields],
                "query": params["query"],
            }
        }
        return score_by, warnings

    if kind == "multi_match_phrase":
        fields = [_parse_boosted_field(f, index, registry) for f in params.get("fields") or []]
        q = str(params["query"]).replace('"', '\\"')
        parts = [f'{fname}:("{q}")' for fname, _ in fields]
        score_by = {"query_string": {"query": " OR ".join(parts)}}
        return score_by, warnings

    if kind == "query_string":
        q = str(params["query"])
        # Rewrite field names inside the query when possible (simple token.field: patterns)
        q = _rewrite_query_string_fields(q, index=index, registry=registry)
        score_by: dict[str, Any] = {"query_string": {"query": q}}
        fields = params.get("fields") or []
        default_field = params.get("default_field")
        if fields:
            rewritten = [registry.rewrite_field(index, _strip_boost(f)) for f in fields]
            score_by["query_string"]["fields"] = rewritten
        elif default_field:
            score_by["query_string"]["fields"] = [registry.rewrite_field(index, default_field)]
        return score_by, warnings

    if kind == "constant_score":
        # No score_by — planner uses fetch with boost
        return {"constant_score": {"boost": params.get("boost", 1.0)}}, warnings

    if kind == "neural":
        field = registry.rewrite_field(index, params["field"])
        score_by = {
            "dense_vector": {
                "field": field,
                "values": None,  # filled by executor after embed
                "top_k": top_k if params.get("k") is None else min(int(params["k"]), top_k),
            },
            "_embed": {
                "model_id": params["model_id"],
                "query_text": params["query_text"],
                "sparse": False,
            },
        }
        return score_by, warnings

    if kind == "neural_sparse":
        field = registry.rewrite_field(index, params["field"])
        score_by = {
            "sparse_vector": {
                "field": field,
                "sparse_values": None,
                "top_k": top_k if params.get("k") is None else min(int(params["k"]), top_k),
            },
            "_embed": {
                "model_id": params["model_id"],
                "query_text": params["query_text"],
                "sparse": True,
            },
        }
        return score_by, warnings

    raise ScoringCompileError(path, f"Unsupported scoring kind {kind!r}")


def _strip_boost(field: str) -> str:
    return field.split("^", 1)[0]


def _parse_boosted_field(raw: str, index: str, registry: MappingRegistry) -> tuple[str, float]:
    if "^" in raw:
        name, boost_s = raw.split("^", 1)
        return registry.rewrite_field(index, name), float(boost_s)
    return registry.rewrite_field(index, raw), 1.0


def _fuzziness_edits(fuzziness: Any, query: str) -> int:
    if fuzziness is None:
        return 0
    if isinstance(fuzziness, int):
        return fuzziness
    if str(fuzziness).upper() == "AUTO":
        # Use longest token heuristic
        tokens = query.split() or [query]
        longest = max(len(t) for t in tokens)
        if longest <= 2:
            return 0
        if longest <= 5:
            return 1
        return 2
    try:
        return int(fuzziness)
    except (TypeError, ValueError):
        return 1


_FIELD_PREFIX = re.compile(r"(?P<field>[A-Za-z_][\w.]*)\:")


def _rewrite_query_string_fields(query: str, *, index: str, registry: MappingRegistry) -> str:
    def repl(match: re.Match[str]) -> str:
        field = match.group("field")
        return f"{registry.rewrite_field(index, field)}:"

    return _FIELD_PREFIX.sub(repl, query)


def and_match_filter_for_clause(
    clause: ScoringClause,
    *,
    index: str,
    registry: MappingRegistry,
) -> dict[str, Any] | None:
    """Extra filter required for match operator:and / msm 100%."""
    if clause.kind != "match":
        return None
    operator = clause.params.get("operator") or "or"
    msm = clause.params.get("minimum_should_match")
    if operator == "and" or str(msm) in ("100%", "100"):
        field = registry.rewrite_field(index, clause.params["field"])
        return {field: {"$match_all": clause.params["query"]}}
    return None

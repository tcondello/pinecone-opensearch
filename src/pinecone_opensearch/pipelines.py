"""Local search_pipeline registry (config + runtime put/get)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class PipelineRegistry:
    def __init__(self, initial: dict[str, Any] | None = None):
        self._pipelines: dict[str, dict[str, Any]] = {
            k: deepcopy(v) for k, v in (initial or {}).items()
        }

    def put(self, pipeline_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if not pipeline_id:
            raise ValueError("pipeline id is required")
        if not isinstance(body, dict):
            raise ValueError("pipeline body must be an object")
        self._pipelines[pipeline_id] = deepcopy(body)
        return {"acknowledged": True}

    def get(self, pipeline_id: str | None = None) -> dict[str, Any]:
        if pipeline_id is None:
            return deepcopy(self._pipelines)
        if pipeline_id not in self._pipelines:
            from opensearchpy.exceptions import NotFoundError

            raise NotFoundError(
                404,
                f"pipeline_not_found_exception: {pipeline_id}",
                {"pipeline": pipeline_id},
            )
        return {pipeline_id: deepcopy(self._pipelines[pipeline_id])}

    def delete(self, pipeline_id: str) -> dict[str, Any]:
        self._pipelines.pop(pipeline_id, None)
        return {"acknowledged": True}

    def resolve(self, pipeline_id: str | None) -> dict[str, Any] | None:
        if not pipeline_id:
            return None
        return self._pipelines.get(pipeline_id)

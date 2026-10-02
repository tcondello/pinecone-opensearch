"""indices.* — create / delete / exists / get_mapping / refresh (M1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import NotFoundError, RequestError

from pinecone_opensearch.config import IndexConfig
from pinecone_opensearch.errors import raise_as_opensearch
from pinecone_opensearch.exceptions import UnsupportedQueryError
from pinecone_opensearch.executor import wait_for_ids
from pinecone_opensearch.writes import translate_os_mappings_to_pinecone_schema

if TYPE_CHECKING:
    from pinecone_opensearch.client import OpenSearch


class IndicesClient:
    def __init__(self, client: OpenSearch) -> None:
        self._client = client

    def create(
        self,
        index: str,
        body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body = body or kwargs.get("body") or {}
        mappings = body.get("mappings") or {}
        client = self._client
        knn_field = None
        if index in client.config.indices:
            knn_field = client.config.indices[index].knn_field

        try:
            schema, warnings = translate_os_mappings_to_pinecone_schema(
                mappings, knn_field=knn_field
            )
        except UnsupportedQueryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RequestError(400, str(exc), {"reason": str(exc)}) from exc

        pinecone_name = index
        if index in client.config.indices:
            pinecone_name = client.config.indices[index].pinecone_index
        else:
            # Register a new logical index pointing at same name
            client.config.indices[index] = IndexConfig(
                pinecone_index=index,
                mapping=mappings if isinstance(mappings, dict) else {},
                knn_field=knn_field,
            )

        # Persist mapping on config + registry
        idx_cfg = client.config.indices[index]
        idx_cfg.mapping = mappings if isinstance(mappings, dict) else {"properties": {}}
        idx_cfg.pinecone_index = pinecone_name
        client.registry.register_index(index, idx_cfg)
        client.registry.store_raw_mapping(index, mappings)

        deployment = dict(client.config.raw.get("deployment") or {})
        if not deployment:
            deployment = {
                "deployment_type": "managed",
                "cloud": "aws",
                "region": "us-east-1",
            }

        try:
            client._backend.pc.indexes.create(
                name=pinecone_name,
                schema=schema,
                deployment=deployment,
            )
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise

        acknowledged = {"acknowledged": True, "index": index, "shards_acknowledged": True}
        if warnings:
            acknowledged["_pinecone"] = {"warnings": warnings}
        return acknowledged

    def delete(self, index: str, **kwargs: Any) -> dict[str, Any]:
        client = self._client
        logical = client.config.aliases.get(index, index)
        pinecone_name = index
        if logical in client.config.indices:
            pinecone_name = client.config.indices[logical].pinecone_index
        try:
            client._backend.pc.indexes.delete(name=pinecone_name)
        except Exception as exc:  # noqa: BLE001
            raise_as_opensearch(exc)
            raise
        client.config.indices.pop(logical, None)
        client.registry.drop_index(logical)
        return {"acknowledged": True}

    def exists(self, index: str, **kwargs: Any) -> bool:
        client = self._client
        logical = client.config.aliases.get(index, index)
        pinecone_name = (
            client.config.indices[logical].pinecone_index
            if logical in client.config.indices
            else index
        )
        try:
            client._backend.pc.indexes.describe(name=pinecone_name)
            return True
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
            if status == 404:
                return False
            # Some SDKs raise NotFoundError without status
            if "not found" in str(exc).lower():
                return False
            raise_as_opensearch(exc)
            raise

    def get_mapping(self, index: str | None = None, **kwargs: Any) -> dict[str, Any]:
        client = self._client
        names: list[str]
        if index is None:
            names = list(client.config.indices)
        else:
            names = [index]
        out: dict[str, Any] = {}
        for name in names:
            logical = client.config.aliases.get(name, name)
            if logical not in client.config.indices:
                raise NotFoundError(404, f"index_not_found_exception: {name}", {"index": name})
            mapping = client.registry.get_raw_mapping(logical)
            if mapping is None:
                mapping = client.config.indices[logical].mapping or {"properties": {}}
            out[logical] = {"mappings": mapping}
        return out

    def refresh(self, index: str | list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        client = self._client
        if index is None:
            targets = list(client.config.indices)
        elif isinstance(index, list):
            targets = index
        else:
            targets = [index]

        shards: dict[str, Any] = {}
        for name in targets:
            logical, index_cfg, ns = client._resolve(name)
            ids = list(client._last_written.get(index_cfg.pinecone_index, []))
            waited = 0.0
            if ids:
                waited = wait_for_ids(
                    client._backend,
                    pinecone_index=index_cfg.pinecone_index,
                    namespace=ns,
                    ids=ids[-1000:],
                    timeout_s=float(client.config.limits.get("refresh_timeout_s", 90)),
                )
            shards[logical] = {
                "total": 1,
                "successful": 1,
                "failed": 0,
                "_pinecone_wait_s": waited,
            }
        return {"_shards": {"total": len(targets), "successful": len(targets), "failed": 0}, **{
            k: {"_shards": v} for k, v in shards.items()
        }}

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(f"indices.{name} is not implemented")

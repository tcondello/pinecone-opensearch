"""Map Pinecone SDK errors to opensearch-py exceptions."""

from __future__ import annotations

from typing import Any

from opensearchpy.exceptions import (
    AuthenticationException,
    AuthorizationException,
    ConflictError,
    NotFoundError,
    RequestError,
    TransportError,
)

try:
    from pinecone.errors.exceptions import (
        ApiError,
        ConflictError as PineconeConflictError,
        ForbiddenError,
        NotFoundError as PineconeNotFoundError,
        RateLimitError,
        UnauthorizedError,
    )
except ImportError:  # pragma: no cover
    ApiError = Exception  # type: ignore[misc, assignment]
    PineconeConflictError = Exception  # type: ignore[misc, assignment]
    ForbiddenError = Exception  # type: ignore[misc, assignment]
    PineconeNotFoundError = Exception  # type: ignore[misc, assignment]
    RateLimitError = Exception  # type: ignore[misc, assignment]
    UnauthorizedError = Exception  # type: ignore[misc, assignment]


def raise_as_opensearch(exc: BaseException) -> None:
    """Re-raise *exc* as the matching opensearch-py exception (never returns)."""
    if isinstance(exc, (NotFoundError, RequestError, ConflictError, AuthenticationException,
                        AuthorizationException, TransportError)):
        raise exc

    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    message = str(exc)
    info: dict[str, Any] = {"type": type(exc).__name__, "reason": message}

    if isinstance(exc, PineconeNotFoundError) or status == 404:
        raise NotFoundError(404, message, info) from exc
    if isinstance(exc, PineconeConflictError) or status == 409:
        raise ConflictError(409, message, info) from exc
    if isinstance(exc, UnauthorizedError) or status == 401:
        raise AuthenticationException(401, message, info) from exc
    if isinstance(exc, ForbiddenError) or status == 403:
        raise AuthorizationException(403, message, info) from exc
    if isinstance(exc, RateLimitError) or status == 429:
        raise TransportError(429, message, info) from exc
    if isinstance(exc, ApiError) or (isinstance(status, int) and 400 <= status < 500):
        raise RequestError(status or 400, message, info) from exc
    raise TransportError(status or 500, message, info) from exc

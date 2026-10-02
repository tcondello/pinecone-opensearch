"""Exceptions compatible with opensearch-py, plus translate-specific errors."""

from opensearchpy.exceptions import RequestError


class UnsupportedQueryError(RequestError):
    """Raised when a query clause cannot be translated faithfully (strict mode)."""

    def __init__(self, path: str, message: str | None = None, status_code: int = 400):
        detail = message or f"Unsupported OpenSearch DSL at {path}"
        error = {
            "type": "unsupported_query_exception",
            "reason": detail,
            "path": path,
        }
        super().__init__(status_code, detail, error)
        self.path = path

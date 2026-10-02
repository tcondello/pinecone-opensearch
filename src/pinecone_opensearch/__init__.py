"""OpenSearch drop-in client backed by the Pinecone Documents API."""

from pinecone_opensearch import helpers as helpers
from pinecone_opensearch.client import OpenSearch
from pinecone_opensearch.exceptions import UnsupportedQueryError
from pinecone_opensearch.shadow import ShadowClient

__all__ = ["OpenSearch", "UnsupportedQueryError", "ShadowClient", "helpers"]
__version__ = "0.1.0"

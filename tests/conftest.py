from pathlib import Path

import pytest

from pinecone_opensearch import OpenSearch


FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "pinecone_opensearch.yaml"


@pytest.fixture
def client() -> OpenSearch:
    return OpenSearch(config=FIXTURE_CONFIG)

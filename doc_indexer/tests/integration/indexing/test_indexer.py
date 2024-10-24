import uuid
from typing import List
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from indexing.indexer import MarkdownIndexer
from src.utils.hana import create_hana_connection
from src.utils.models import create_embedding_factory, openai_embedding_creator
from src.utils.settings import (
    EMBEDDING_MODEL_DEPLOYMENT_ID,
    DATABASE_URL,
    DATABASE_PORT,
    DATABASE_USER,
    DATABASE_PASSWORD,
)

docs_path = "./integration/fixtures/test_docs"


@pytest.fixture(scope="session")
def embedding_model():
    create_embedding = create_embedding_factory(openai_embedding_creator)
    embedding_model = create_embedding(EMBEDDING_MODEL_DEPLOYMENT_ID)
    return embedding_model


@pytest.fixture(scope="session")
def hana_conn():
    # setup connection to Hana Cloud DB
    hana_conn = create_hana_connection(
        DATABASE_URL, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD
    )
    yield hana_conn
    hana_conn.close()


@pytest.fixture(scope="session")
def table_name():
    return f"test_table_{uuid.uuid4().hex}"


@pytest.fixture(scope="session")
def indexer(embedding_model, hana_conn, table_name):
    indexer = MarkdownIndexer(
        docs_path, embedding_model, hana_conn, table_name=table_name
    )
    yield indexer
    try:
        indexer.db.drop_table()
    except Exception as e:
        print(f"Error while dropping table: {e}")


@pytest.fixture
def loaded_documents() -> List[Document]:
    return [
        Document(
            page_content="""
            # Istio management
            Content 0""".strip(),
            metadata={"source": "doc-1"},
            type="Document",
        ),
        Document(
            page_content="""
            # Header section
            Content 1
            ## Header two
            Content 2
            # Header one
            Content 3""".strip(),
            metadata={"source": "doc-2"},
            type="Document",
        ),
        Document(
            page_content="""
            # Another Header
            Some content
            ## Subheader
            More content""".strip(),
            metadata={"source": "doc-3"},
            type="Document",
        ),
        Document(
            page_content="No headers here, just plain content.",
            metadata={"source": "doc-4"},
            type="Document",
        ),
    ]


@pytest.mark.parametrize(
    "test_case, headers_to_split_on, expected_chunks",
    [
        (
            "Split with '#' Header 1",
            [("#", "Header1")],
            [
                "# Istio management\nContent 0",
                "# Header section\nContent 1\n## Header two\nContent 2",
                "# Header one\nContent 3",
                "# Another Header\nSome content\n## Subheader\nMore content",
                "No headers here, just plain content.",
            ],
        ),
        (
            "Split with '#' Header 1 and '##' Header 2",
            [("#", "Header1"), ("##", "Header2")],
            [
                "# Istio management\nContent 0",
                "# Header section\nContent 1",
                "## Header two\nContent 2",
                "# Header one\nContent 3",
                "# Another Header\nSome content",
                "## Subheader\nMore content",
                "No headers here, just plain content.",
            ],
        ),
    ],
)
def test_index(
    test_case,
    headers_to_split_on,
    expected_chunks,
    hana_conn,
    table_name,
    loaded_documents,
    indexer,
):
    with patch.object(indexer, "_load_documents", return_value=loaded_documents):
        indexer.index(headers_to_split_on)
        cursor = hana_conn.cursor()
        cursor.execute(f'SELECT VEC_TEXT, VEC_META FROM "{table_name}"')
        stored_chunks = [row[0] for row in cursor.fetchall()]
        cursor.close()

        for chunk in expected_chunks:
            assert chunk in stored_chunks, f"Expected chunk not found: {chunk}"

        assert len(stored_chunks) == len(expected_chunks), (
            f"Number of chunks mismatch. Expected {len(expected_chunks)}, "
            f"but got {len(stored_chunks)}"
        )
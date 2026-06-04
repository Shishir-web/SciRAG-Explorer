import pytest
from unittest.mock import patch, MagicMock

# Import chunk_models first so SQLAlchemy registers both models together
import db.chunk_models  # noqa: F401
from ingestion.arxiv_client import fetch_arxiv

MOCK_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Test Paper on Neuroinflammation</title>
    <summary>This paper investigates GLP-1 effects on neuroinflammation.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Jane Smith</name></author>
    <link title="pdf" href="https://arxiv.org/pdf/2401.00001v1"/>
  </entry>
</feed>"""


def test_fetch_arxiv_parses_correctly():
    mock_resp = MagicMock()
    mock_resp.text = MOCK_ARXIV_XML
    mock_resp.raise_for_status = lambda: None

    with patch("ingestion.arxiv_client.requests.get", return_value=mock_resp):
        papers = list(fetch_arxiv("GLP-1 neuroinflammation", max_results=1))

    assert len(papers) == 1
    p = papers[0]
    assert p.source == "arxiv"
    assert p.id == "arxiv:2401.00001v1"
    assert "Neuroinflammation" in p.title
    assert "Jane Smith" in p.authors
    assert p.abstract.startswith("This paper")


def test_paper_id_format():
    """IDs must be namespaced to avoid collisions between sources."""
    mock_resp = MagicMock()
    mock_resp.text = MOCK_ARXIV_XML
    mock_resp.raise_for_status = lambda: None

    with patch("ingestion.arxiv_client.requests.get", return_value=mock_resp):
        papers = list(fetch_arxiv("test", max_results=1))

    assert papers[0].id.startswith("arxiv:")
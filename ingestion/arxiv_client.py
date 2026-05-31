import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterator
from db.models import Paper

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}


def fetch_arxiv(query: str, max_result: int = 100) -> Iterator[Paper]:

    """Fetch papers from arXiv API based on the query and yield Paper objects."""
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_result,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)

    for entry in root.findall("atom:entry", NS):
        arxiv_id = entry.find("arxiv:id", NS).text.split("/abs/")[-1]
        title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", NS).text.strip()
        published_raw = entry.find("atom:published", NS).text
        published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))

        authors = [
            a.find("atom:name", NS).text
            for a in entry.findall("atom:author", NS)
        ]

        pdf_url = next(
            (l.get("href") for l in entry.findall("atom:link",NS)
             if l.get("title") == "pdf"), 
            None
        )

        doi_el = entry.find("arxiv:doi", NS)
        doi = doi_el.text if doi_el is not None else None
        
        yield Paper(
            id      = f"arxiv:{arxiv_id}",
            source  = "arxiv",
            title   = title,
            abstract = abstract,
            authors = authors,
            published= published,
            doi     = doi,
            url     = pdf_url,
            raw_meta = {"arxiv_id": arxiv_id}
        )
        
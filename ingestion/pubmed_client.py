import os
from Bio import Entrez
from datetime import datetime
from typing import Iterator
from db.models import Paper


def fetch_pubmed(query: str, max_results: int = 100) -> Iterator[Paper]:
    """
    Fetch papers from PubMed via Biopython's Entrez wrapper.
    Yields Paper ORM objects.
    Skips gracefully if NCBI_EMAIL is not set.
    """
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        print("  NCBI_EMAIL not set — skipping PubMed ingestion")
        return

    Entrez.email   = email
    Entrez.api_key = os.environ.get("NCBI_API_KEY")

    try:
        handle = Entrez.esearch(db="pubmed", term=query,
                                retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        pmids  = record["IdList"]

        if not pmids:
            return

        handle  = Entrez.efetch(db="pubmed", id=",".join(pmids),
                                rettype="xml", retmode="xml")
        records = Entrez.read(handle)

    except Exception as e:
        print(f"  PubMed request failed: {e} — skipping")
        return

    for article in records["PubmedArticle"]:
        medline  = article["MedlineCitation"]
        art_data = medline["Article"]

        pmid  = str(medline["PMID"])
        title = str(art_data.get("ArticleTitle", ""))

        abs_texts = art_data.get("Abstract", {}).get("AbstractText", [])
        abstract  = " ".join(str(t) for t in abs_texts)

        authors = []
        for a in art_data.get("AuthorList", []):
            name = f"{a.get('ForeName','')} {a.get('LastName','')}".strip()
            if name:
                authors.append(name)

        pub_date = medline.get("DateCompleted") or {}
        year  = pub_date.get("Year",  "1900")
        month = pub_date.get("Month", "1")
        day   = pub_date.get("Day",   "1")
        try:
            published = datetime(int(year), int(month), int(day))
        except ValueError:
            published = datetime(int(year), 1, 1)

        yield Paper(
            id       = f"pubmed:{pmid}",
            source   = "pubmed",
            title    = title,
            abstract = abstract,
            authors  = authors,
            published= published,
            url      = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            raw_metadata = {"pmid": pmid},
        )
import os
from Bio import Entrez
from datetime import datetime 
from typing import Iterator
from db.models import Paper

def fetch_pubmed(query: str, max_results: int = 100) -> Iterator[Paper]:
    """ Fetch papers from PubMed via Biopython's Entrez wrapper.
    Yields Paper ORM objects.
    """
    Entrez.email = os.environ["NCBI_EMAIL"]
    Entrez.api_key = os.environ.get("NCBI_API_KEY")  # Optional, but recommended for higher rate limits

    # Step 1: Search for PMIDs matching the query
    handle = Entrez.esearch(db="pubmed", term=query,
                            retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    pmids = record["IdList"]

    if not pmids:
        return
    
    # Step 2: Fetch details for each PMID
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids),
                           rettype="xml", retmode="xml")
    records = Entrez.read(handle)

    for article in records["PubmedArticle"]:
        medline = article["MedicineCitation"]
        art_data = medline["Article"]

        pmid = str(medline["PMID"])
        title = (art_data.get("ArticleTitle", ""))

        #abstract can be a list of sections or a single string
        abs_texts = art_data.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(t) for t in abs_texts)

        authors = []
        for a in art_data.get("AuthorList", []):
            name = f"{a.get('Forename', '')} {a.get('LastName', '')}".strip()
            if name:
                authors.append(name)

        pub_date = medline.get("DateCompleted", {})
        year = pub_date.get("Year", "1900")
        month = pub_date.get("Month", "01")
        day = pub_date.get("Day", "01")
        try:
            published = datetime(int(year), int(month), int(day))
        except ValueError:
            published = datetime(int(year), 1, 1)

        yield Paper(
            id = f"pubmed:{pmid}",
            source = "pubmed",
            title = title,
            abstract = abstract,
            authors = authors,
            published = published,
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            raw_meta = {"pmid": pmid}
        )
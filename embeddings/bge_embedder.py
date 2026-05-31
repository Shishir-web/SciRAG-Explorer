import os
import time
import requests
import numpy as np
from typing import List

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BATCH_SIZE = 32
RETRY_WAIT = 5 # seconds on 503

def get_headers() -> dict:
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        raise ValueError("HF_API_TOKEN not set in environment")
    return {"Authorization": f"Bearer {token}"}

def embed_batch(texts:  list[str]) -> list[list[float]]:
    """Call HF Inference for a batch of texts.
       Returns list of 1024-dim float vectors. 
    """
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True, "use_cache": True},
    }

    for attempt in range(3):
        resp = requests.post(HF_API_URL, headers=get_headers(),
                             json=payload, timeout=60)

        if resp.status_code == 503:
            # Model is still loading on HF side
            print(f"Model not ready yet. Retrying in {RETRY_WAIT} seconds...")
            time.sleep(RETRY_WAIT)
            continue

        resp.rate_for_status()
        result = resp.json()

        #HF returns shape [batch, seq_len, dim] or [batch, dim]
        #BGE-M3 returns shape [batch, dim] takr first token (CLS) if nested
        vectors = []
        for item in result:
            if isinstance(item[0], list):
                #Nested: take mean pooling
                vec = np.mean(item, axis=0).tolist()
            else:
                vec = item
            vectors.append(vec)

        return vectors
    
    raise RuntimeError("HF API unavailable after 3 retries")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed an arbitary number of texts, batching automatically.
    """
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f" Emedding batch {i//BATCH_SIZE + 1}"
              f"({len(batch)} texts)...")
        vectors = embed_batch(batch)
        all_vectors.extend(vectors)

    return all_vectors
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

DATASET_PATH = Path(__file__).parent.parent / "data" / "golden_dataset.json"

@dataclass
class GoldenSample:
    id:                     str
    query:                  str
    reference_answer:       str
    ground_truth_paper_ids: list[str]
    expected_has_conflit:   bool
    topic:                  str
    difficulty:             Literal["easy", "medium", "hard"]

def load_golden_dataset(path: Path = DATASET_PATH) -> list[GoldenSample]:
    """Load and validate the golden dataset from JSON."""
    with open(path) as f:
        raw = json.load(f)
    
    samples = []
    for item in raw:
        # validate required fields
        required = [
            "id", "query", "reference_answer",
            "ground_truth_paper_id", "expected_has_conflict",
            "topic", "difficulty",
        ]
        missing = [k for k in required if k not in item]
        if missing:
            raise ValueError(f"Sample{item.get('id', '?')} missing: {missing}")
        
        samples.append(GoldenSample(**item))

    return samples

def filter_by_topic(samples: list[GoldenSample],
                    topic: str) -> list[GoldenSample]:
    return [s for s in samples if s.topic == topic]

def filter_by_difficulty(samples: list[GoldenSample],
                         difficulty: str) ->  list[GoldenSample]:
    return [s for s in samples if s.difficulty == difficulty]


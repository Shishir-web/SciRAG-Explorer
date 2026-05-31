import pytest
from chunking.section_splitter import(
    chunk_paper, detect_section, split_into_sections
)

SAMPLE_ABSTRACT = (
    "This study investigates the role of GLP-1 receptor agonists in "
    "reducing neuroinflammatory markers. We recruited 120 participants "
    "and measured cytokine levels pre and post treatment. Results showed "
    "a significant reduction in IL-6 and TNF-alpha after 12 weeks."
)

SAMPLE_FULL_TEXT = """Abstract
This study investigates GLP-1 receptor agonists and neuroinflammation.

Introduction
Neuroinflammation is a hallmark of many neurodegenerative diseases.
Recent studies suggest GLP-1 receptor agonists may reduce inflammatory markers.

Methods
Participants were randomised into two groups.
Blood samples were collected at baseline and week 12.
Cytokine levels were measured using ELISA.

Results
IL-6 levels decreased by 34% in the treatment group.
TNF-alpha showed a significant reduction (p < 0.01).

Conclusion
GLP-1 receptor agonists show promise in reducing neuroinflammation.
"""


def test_defect_section_label():
    assert detect_section("Abstract")      == "abstract"
    assert detect_section("METHODS")       == "methods" 
    assert detect_section("Materials and Methods")      == "methods"
    assert detect_section("Results")       == "results"
    assert detect_section("Abstract") is None


def test_split_into_sections():
    sections = dict(split_into_sections(SAMPLE_FULL_TEXT))
    assert "introduction" in sections
    assert "methods"      in sections
    assert "results"      in sections
    assert "conclusion"   in sections
    # References should be excluded 
    assert "references"   not in sections

def test_chunk_paper_abstract_always_present():
    chunks = chunk_paper("test:001", SAMPLE_ABSTRACT, None)
    assert len(chunks) >= 1
    assert all(c.section == "abstract" for c in chunks)

def test_chunk_token_limit():
    """No chunk should exceed MAX_TOKENS"""
    from chunking.section_splitter import MAX_TOKENS
    import tiktoken
    enc    = tiktoken.get_encoding("cl100k_base")
    chunks = chunk_paper("test:002", SAMPLE_ABSTRACT, SAMPLE_FULL_TEXT, None)
    for c in chunks:
        assert len(enc.encode(c.text)) <= MAX_TOKENS + 10

def test_chunk_indices_are_sequential():
    chunks = chunk_paper("test:001", SAMPLE_ABSTRACT, SAMPLE_FULL_TEXT)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


    
    

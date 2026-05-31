import re
import tiktoken
from dataclasses import dataclass
from typing import List

ENCODER = tiktoken.get_encoding("cl100k_base")

# Section header patterns to look for (case-insensitive)
SECTION_PATTERNS = [
    (r"(?i)^\s*abstract\b",        "abstract"),
    (r"(?i)^\s*introduction\b",    "introduction"),
    (r"(?i)^\s*background\b",      "background"),
    (r"(?i)^\s*methods?\b",        "methods"),
    (r"(?i)^\s*materials?\s+and\s+methods?\b", "methods"),
    (r"(?i)^\s*results?\b",        "results"),
    (r"(?i)^\s*discussion\b",      "discussion"),
    (r"(?i)^\s*conclusion\b",      "conclusion"),
    (r"(?i)^\s*references?\b",     "references"),
]

MAX_TOKENS = 400 # max tokens per chunk
OVERLAP_CHARS = 200 # character overlap between chunks to preserve context

@dataclass
class TextChunk:
    text:         str
    section:      str
    char_start:   int
    char_end:     int
    token_count:  int
    chunk_index:  int

def detect_section(line: str) -> str | None:
    """Return section label if a line looks like a section header"""
    for pattern, label in SECTION_PATTERNS:
        if re.match(pattern, line.strip()):
            return label
    return None

def split_into_chunks(text: str) -> List[tuple[str, str]]:
    """Split raw text into (section_label, section_text) pairs.
    Falls back to 'body' for unlabelled content. """

    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_section = "body"
    current_lines: list[str] = []

    for line in lines:
        detected = detect_section(line)
        if detected and detected != "references":
            # Save the previous section
            if current_lines:
                sections.append((current_section, "\n".join(current_lines).strip()))
            current_section = detected
            current_lines = []
        elif detected == "references":
            # Skip the references section
            break
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, "\n".join(current_lines).strip()))

    return [(s, t) for s, t in sections if t]

def chunk_section(section: str, text: str, char_offset: int,
                  start_index: int) -> List[TextChunk]:
    
    """Split a single section into token-bounded chunks with overlap."""

    chunks: list[TextChunk] = []
    words = text.split(" ")
    buffer: list[str] = []
    buf_start_char = char_offset
    chunk_idx = start_index

    def flush(buf: list[str], start: int) -> TextChunk | None:
        joined = " ".join(buf).strip()
        if not joined:
            return None
        tokens = len(ENCODER.encode(joined))
        end    = start + len(joined)
        return TextChunk(
            text=joined, section=section, char_start=start,
            char_end=end, token_count=tokens, chunk_index=chunk_idx
        )
    
    pos = char_offset
    for word in words:
        buffer.append(word)
        pos += len(word) + 1 # account for space

        token_count = len(ENCODER.encode(" ".join(buffer)))
        if token_count >= MAX_TOKENS:
            chunk = flush(buffer, buf_start_char)
            if chunk:
                chunks.append(chunk)
                chunk_idx += 1

            # Overlap: keep the last OVERLAP_CHARS characters for the next chunk
            overlap_text = " ".join(buffer)[-OVERLAP_CHARS:]
            buffer = overlap_text.split(" ")
            buf_start_char = pos - len(overlap_text)

    # Flush any remaining buffer
    chunk = flush(buffer, buf_start_char)
    if chunk:
        chunks.append(chunk)

    return chunks

def chunk_paper(paper_id: str, abstract: str,
                full_text: str | None) -> List[TextChunk]:
    """ Main entry point: given a paper's abstract and full text, split into labeled chunks. 
    If full_text is None, will chunk the abstract only. """

    all_chunks: list[TextChunk] = []
    idx = 0

    # Always include the abstract as the first chunk
    if abstract:
        for chunk in chunk_section("abstract", abstract, 0, start_index=idx):
            all_chunks.append
            idx += 1

    if full_text:
        sections = split_into_chunks(full_text)
        char_offset = 0

        for section_label, section_text in sections:
            if section_label == "abstract":
                #Already processed the abstract, skip it in the full text
                char_offset += len(section_text)
                continue
            new_chunks = chunk_section(section_label, section_text, 
                                       char_offset, idx)
            all_chunks.extend(new_chunks)
            idx += len(new_chunks)
            char_offset += len(section_text)

    return all_chunks

import os
import re

# Matches: "4. Title", "4.1 Title", "4.1.1 Title"
_HEADER_RE = re.compile(
    r'^(\d+\.\d+\.\d+|\d+\.\d+|\d+\.)\s+[A-Z].+$',
    re.MULTILINE,
)

MAX_WORDS = 350   # soft ceiling before further splitting
OVERLAP_WORDS = 60  # word overlap when falling back to word-based splitting


def _strip_page_markers(text: str) -> str:
    return re.sub(r'--- Page \d+ ---\n?', '', text)


def _word_split(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Fixed-size word splitting with overlap — used as a last resort."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap
    return chunks


def _split_oversized(section: str) -> list[str]:
    """
    Break a section that exceeds MAX_WORDS.
    Strategy: paragraph boundaries first, word-split as fallback.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\n+', section) if p.strip()]
    chunks, bucket = [], []

    for para in paragraphs:
        candidate = bucket + [para]
        if len(" ".join(candidate).split()) <= MAX_WORDS:
            bucket = candidate
        else:
            if bucket:
                chunks.append("\n\n".join(bucket))
            # Para itself may be oversized
            if len(para.split()) > MAX_WORDS:
                chunks.extend(_word_split(para))
                bucket = []
            else:
                bucket = [para]

    if bucket:
        chunks.append("\n\n".join(bucket))

    return chunks


def semantic_chunk(text: str) -> list[str]:
    """
    Split on numbered section/sub-section headers found in the document.
    Each chunk = header line + its body text.
    Sections exceeding MAX_WORDS are recursively split on paragraphs.
    Falls back to word-based splitting if no headers are found.
    """
    text = _strip_page_markers(text)

    positions = [(m.start(), m.end()) for m in _HEADER_RE.finditer(text)]

    if not positions:
        # No headers — plain paragraph split then word-split if needed
        raw = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        result = []
        for para in raw:
            if len(para.split()) > MAX_WORDS:
                result.extend(_word_split(para))
            else:
                result.append(para)
        return result

    sections = []
    for i, (h_start, _) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section = text[h_start:end].strip()
        if section:
            sections.append(section)

    result = []
    for section in sections:
        if len(section.split()) <= MAX_WORDS:
            result.append(section)
        else:
            result.extend(_split_oversized(section))

    return [c for c in result if c.strip()]


def save_chunks(chunks: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(output_dir):
        if f.startswith("chunk_") and f.endswith(".txt"):
            os.remove(os.path.join(output_dir, f))

    for i, chunk in enumerate(chunks):
        with open(os.path.join(output_dir, f"chunk_{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk)

    print(f"[chunker] Saved {len(chunks)} semantic chunks to {output_dir}")

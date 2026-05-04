"""
Tokenizer-aware text chunking for augmentation models.

The helpers keep long emails from being silently truncated by models with
short context windows. They split on paragraphs and sentences first, falling
back to token slices only when a single segment is too long.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    text: str
    token_count: int
    forced_split: bool = False


def token_count(tokenizer: Any, text: str) -> int:
    """Counts tokenizer tokens without adding model special tokens."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def chunk_text(
    text: str,
    tokenizer: Any,
    max_tokens: int,
) -> list[TextChunk]:
    """
    Splits one text into chunks that fit max_tokens.

    The function does not truncate. If a sentence is longer than max_tokens,
    it is split by tokenizer ids and marked as forced_split.
    """
    clean_text = str(text).strip()
    if not clean_text:
        return []

    if token_count(tokenizer, clean_text) <= max_tokens:
        return [TextChunk(clean_text, token_count(tokenizer, clean_text))]

    segments = _split_into_segments(clean_text)
    chunks: list[TextChunk] = []
    current: list[str] = []
    current_tokens = 0

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        seg_tokens = token_count(tokenizer, segment)
        if seg_tokens > max_tokens:
            if current:
                chunks.append(TextChunk(_join_segments(current), current_tokens))
                current = []
                current_tokens = 0
            chunks.extend(_split_long_segment(segment, tokenizer, max_tokens))
            continue

        separator_tokens = token_count(tokenizer, "\n\n") if current else 0
        projected = current_tokens + separator_tokens + seg_tokens

        if current and projected > max_tokens:
            chunks.append(TextChunk(_join_segments(current), current_tokens))
            current = [segment]
            current_tokens = seg_tokens
        else:
            current.append(segment)
            current_tokens = projected

    if current:
        chunks.append(TextChunk(_join_segments(current), current_tokens))

    return chunks


def join_chunks(chunks: list[str]) -> str:
    """Joins generated chunks back into one email-like text."""
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()


def _split_into_segments(text: str) -> list[str]:
    segments: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()]
        segments.extend(sentences if len(sentences) > 1 else [paragraph])
    return segments


def _split_long_segment(
    segment: str,
    tokenizer: Any,
    max_tokens: int,
) -> list[TextChunk]:
    ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
    chunks: list[TextChunk] = []
    for start in range(0, len(ids), max_tokens):
        part_ids = ids[start:start + max_tokens]
        part_text = tokenizer.decode(
            part_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if part_text:
            chunks.append(TextChunk(part_text, len(part_ids), forced_split=True))
    return chunks


def _join_segments(segments: list[str]) -> str:
    return "\n\n".join(s.strip() for s in segments if s.strip()).strip()

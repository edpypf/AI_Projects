from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Callable
import re
@dataclass
class Chunk:
    id: int
    text: str
    start_char: int
    end_char: int
    
def read_text_file(path: str|Path) -> str:
	p=Path(path)
	if not p.exists():
		raise FileNotFoundError(f"File {p} does not exist.")
	text = p.read_text(encoding='utf-8', errors='ignore')
	return text.strip()

def split_sentences(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    if not text:
        return []
    "Simple sentence boundary on .,!. ? while keeping indices"
    pattern = re.compile(r"(?<=[.!?])\s+")
    parts = pattern.split(text)
    spans: List[Tuple[str,Tuple[int, int]]] = []	
    cursor = 0
    for part in parts:
        start = text.find(part, cursor)
        end = start + len(part)
        spans.append((part.strip(), (start, end)))
        cursor = end
    return spans

def window_words(words: list[str], size: int, overlap: int) -> Iterable[Tuple[int, int]]:
    if size <=0:
        raise ValueError("Size must be greater than 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must be >= 0 and < size-1")
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield (start, end)
        if end == len(words):
            break
        start = end - overlap

def fixed_size_chunk(text: str, size_words: int = 200, overlap_words: int = 40) -> List[Chunk]:
	if not text:
		return []
	words = text.split()
	chunks: List[Chunk] = []
	for i, (w_start, w_end) in enumerate(window_words(words, size_words, overlap_words)):
		chunk_words = words[w_start:w_end]
		chunk_text = " ".join(chunk_words)
		start_char = text.find(chunk_text)  # approximate
		end_char = start_char + len(chunk_text)
		chunks.append(Chunk(i, chunk_text, start_char, end_char))
	return chunks
     
def semantic_chunk(text: str, target_chars: int = 800, max_chars: int = 1200) -> List[Chunk]:
	if not text:
		return []
	sentences = split_sentences(text)
	if not sentences:
		return [Chunk(0, text, 0, len(text))]
	chunks: List[Chunk] = []
	current: List[str] = []
	current_start = sentences[0][1][0]
	for sent, (s_start, s_end) in sentences:
		candidate = (" ".join(current) + (" " if current else "") + sent)
		if len(candidate) <= max_chars:
			current.append(sent)
			continue
		# finalize current
		chunk_text = " ".join(current).strip()
		if chunk_text:
			chunks.append(Chunk(len(chunks), chunk_text, current_start, current_start + len(chunk_text)))
		# reset with current sentence
		current = [sent]
		current_start = s_start
	# flush
	chunk_text = " ".join(current).strip()
	if chunk_text:
		chunks.append(Chunk(len(chunks), chunk_text, current_start, current_start + len(chunk_text)))
	# second pass: try to split oversized by naive midpoint if needed
	refined: List[Chunk] = []
	for ch in chunks:
		if len(ch.text) <= max_chars:
			refined.append(Chunk(len(refined), ch.text, ch.start_char, ch.end_char))
			continue
		mid = len(ch.text) // 2
		refined.append(Chunk(len(refined), ch.text[:mid].rstrip(), ch.start_char, ch.start_char + mid))
		refined.append(Chunk(len(refined), ch.text[mid:].lstrip(), ch.start_char + mid, ch.end_char))
	return refined

def mixed_chunk(text: str, target_chars: int = 800, fallback_words: int = 200, overlap_words: int = 40) -> List[Chunk]:
	chunks = semantic_chunk(text, target_chars=target_chars, max_chars=int(target_chars * 1.5))
	if chunks:
		return chunks
	return fixed_size_chunk(text, size_words=fallback_words, overlap_words=overlap_words)
     
def chunk_stats(chunks: List[Chunk]) -> dict:
	if not chunks:
		return {"num_chunks": 0, "avg_chars": 0, "avg_words": 0}
	lengths = [len(c.text) for c in chunks]
	word_counts = [len(c.text.split()) for c in chunks]
	return {
		"num_chunks": len(chunks),
		"avg_chars": sum(lengths) / len(lengths),
		"avg_words": sum(word_counts) / len(word_counts),
	}

def jaccard_similarity(a: set[str], b: set[str]) -> float:
	if not a and not b:
		return 1.0
	if not a or not b:
		return 0.0
	return len(a & b) / len(a | b)

def coherence_score(chunk: Chunk) -> float:
	# crude proxy: measure overlap of unique words between first and second half
	text = chunk.text
	if not text:
		return 0.0
	mid = max(1, len(text) // 2)
	first = set(re.findall(r"[A-Za-z0-9]+", text[:mid].lower()))
	second = set(re.findall(r"[A-Za-z0-9]+", text[mid:].lower()))
	return jaccard_similarity(first, second)

def evaluate_chunks(chunks: List[Chunk]) -> dict:
	stats = chunk_stats(chunks)
	coherence = [coherence_score(c) for c in chunks] if chunks else []
	stats["avg_coherence"] = sum(coherence) / len(coherence) if coherence else 0.0
	return stats

def test_window_words():
	words = [str(i) for i in range(10)]
	spans = list(window_words(words, size=4, overlap=1))
	assert spans == [(0,4),(3,7),(6,10)]


def test_split_sentences():
	text = "A. B? C! D"
	spans = split_sentences(text)
	assert len(spans) == 4
	assert spans[0][0].strip() == "A."
	assert spans[1][0].strip() == "B?"


def test_fixed_size_chunk():
	text = "one two three four five six seven eight nine ten"
	chunks = fixed_size_chunk(text, size_words=3, overlap_words=1)
	assert len(chunks) == 5
	assert chunks[0].text.split() == ["one","two","three"]
	assert chunks[1].text.split() == ["three","four","five"]


def test_semantic_chunk_small():
	text = "A short sentence. Another one. And the last!"
	chunks = semantic_chunk(text, target_chars=20, max_chars=30)
	assert len(chunks) >= 2


def run_tests():
	test_window_words()
	test_split_sentences()
	test_fixed_size_chunk()
	test_semantic_chunk_small()
	print("All tests passed")

run_tests()

TEXT_PATH = f"C:\\works\\ai\\rag\\my_text_file.txt"

try:
	text = read_text_file(TEXT_PATH)
	print(f"Loaded {len(text)} chars from {TEXT_PATH}")
except FileNotFoundError as e:
	print(str(e))
	text = ""

text[:500]

# Generate chunks for each strategy

if text:
	fixed_chunks = fixed_size_chunk(text, size_words=180, overlap_words=40)
	semantic_chunks = semantic_chunk(text, target_chars=900, max_chars=1350)
	mixed_chunks = mixed_chunk(text, target_chars=900, fallback_words=180, overlap_words=40)
	print({
		"fixed": chunk_stats(fixed_chunks),
		"semantic": chunk_stats(semantic_chunks),
		"mixed": chunk_stats(mixed_chunks)
	})
else:
	fixed_chunks = []
	semantic_chunks = []
	mixed_chunks = []
	print("No text loaded; stats unavailable")

# Evaluate chunk quality via crude coherence metric

if text:
	fixed_eval = evaluate_chunks(fixed_chunks)
	semantic_eval = evaluate_chunks(semantic_chunks)
	mixed_eval = evaluate_chunks(mixed_chunks)
	print({
		"fixed": fixed_eval,
		"semantic": semantic_eval,
		"mixed": mixed_eval
	})
else:
	print("No text loaded; evaluation unavailable")

# Preview a few chunks from each strategy

def preview(chunks: List[Chunk], n: int = 2):
	for c in chunks[:n]:
		print(f"[id={c.id}] chars={len(c.text)} coherence={coherence_score(c):.3f}")
		print(c.text[:300].replace("\n"," "))
		print("-"*60)

if text:
	print("Fixed-size preview:")
	preview(fixed_chunks)
	print("Semantic preview:")
	preview(semantic_chunks)
	print("Mixed preview:")
	preview(mixed_chunks)
else:
	print("No text loaded; preview unavailable")
	
# Re-tune parameters to exaggerate differences and regenerate

if text:
	# Smaller windows for clearer contrast on short docs
	FIXED_SIZE_WORDS = 40
	FIXED_OVERLAP_WORDS = 10
	SEM_TARGET_CHARS = 250
	SEM_MAX_CHARS = 350
	MIX_TARGET_CHARS = SEM_TARGET_CHARS
	MIX_FALLBACK_WORDS = FIXED_SIZE_WORDS
	MIX_OVERLAP_WORDS = FIXED_OVERLAP_WORDS

	fixed_chunks = fixed_size_chunk(text, size_words=FIXED_SIZE_WORDS, overlap_words=FIXED_OVERLAP_WORDS)
	semantic_chunks = semantic_chunk(text, target_chars=SEM_TARGET_CHARS, max_chars=SEM_MAX_CHARS)
	mixed_chunks = mixed_chunk(text, target_chars=MIX_TARGET_CHARS, fallback_words=MIX_FALLBACK_WORDS, overlap_words=MIX_OVERLAP_WORDS)

	print({
		"fixed": chunk_stats(fixed_chunks),
		"semantic": chunk_stats(semantic_chunks),
		"mixed": chunk_stats(mixed_chunks)
	})
else:
	print("No text loaded; skip retune")

# Show more chunks with boundaries and indexes

def preview_with_bounds(title: str, chunks: List[Chunk], n: int = 5):
	print(title)
	for c in chunks[:n]:
		print(f"[id={c.id}] span=({c.start_char},{c.end_char}) words={len(c.text.split())} coherence={coherence_score(c):.3f}")
		print(c.text[:220].replace("\n"," "))
		print("-"*60)

if text:
	preview_with_bounds("Fixed-size:", fixed_chunks)
	preview_with_bounds("Semantic:", semantic_chunks)
	preview_with_bounds("Mixed:", mixed_chunks)
else:
	print("No text loaded; preview unavailable")

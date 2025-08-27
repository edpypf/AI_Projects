import os, json, re, glob
from langdetect import detect
from trafilatura import extract
from datasketch import MinHash, MinHashLSH
from collections import Counter

def read_all_texts():
    texts = []

    # Task 1: arxiv_clean.json
    with open("arxiv_clean.json", encoding="utf-8") as f:
        data = json.load(f)
        texts += [item['abstract'] for item in data]

    # Task 2: PDF OCR TXT 檔
    for txt_file in glob.glob("pdf_ocr/*.txt"):
        with open(txt_file, encoding="utf-8") as f:
            texts.append(f.read())

    # Task 3: ASR JSONL
    with open("talks_transcripts.jsonl", encoding="utf-8") as f:
        texts += [json.loads(line)['asr_text'] for line in f if 'asr_text' in json.loads(line)]

    return texts

def is_english(text):
    try:
        return detect(text.strip()) == "en"
    except:
        return False

def remove_pii(text):
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '[EMAIL]', text)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CREDITCARD]', text)
    return text

def has_repetitive_ngrams(text, n=5, threshold=3):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    return any(count >= threshold for count in counts.values())

def minhash_filter(texts, threshold=0.7):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    for i, text in enumerate(texts):
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        if not list(lsh.query(m)):
            lsh.insert(f"doc_{i}", m)
            unique_texts.append(text)
    return unique_texts

def clean_text(text):
    text = extract(text) or text  # 清洗 HTML
    text = remove_pii(text)
    text = text.strip()
    return text

# 主流程
texts = read_all_texts()
original_token_count = sum(len(t.split()) for t in texts)

# 處理流程
texts = [t for t in texts if t.strip()]
texts = [clean_text(t) for t in texts if is_english(t)]
texts = [t for t in texts if not has_repetitive_ngrams(t)]
texts = minhash_filter(texts)

# 儲存結果
with open("clean_corpus.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t.strip() + "\n\n")

# 統計資訊
final_token_count = sum(len(t.split()) for t in texts)
removal_pct = 100 * (1 - final_token_count / original_token_count)

with open("stats.md", "w", encoding="utf-8") as f:
    f.write(f"# 🧹 Cleaning Stats\n\n")
    f.write(f"- Total original tokens: {original_token_count}\n")
    f.write(f"- Final cleaned tokens: {final_token_count}\n")
    f.write(f"- Removal percentage: {removal_pct:.2f}%\n")
    f.write(f"- Final text count: {len(texts)}\n")

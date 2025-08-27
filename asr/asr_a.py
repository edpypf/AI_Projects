import whisper
from faster_whisper import WhisperModel
from datasketch import MinHashLSH, minhash  
from langdetect import detect
from bs4 import BeautifulSoup
import re
from collections import Counter



def using_whisper():
    model = whisper.load_model("base")
    result = model.transcribe("temp.wav")
    print(result["text"])


def using_faster_whisper():
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe("temp.wav", beam_size=5, language="en")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

def minhash_dedup(texts, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    for i, text in enumerate(texts):
        m = minhash.MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        if not lsh.query(m):
            lsh.insert(i, m)
            unique_texts.append(text)
    return unique_texts

def clean_html_and_filter_lang(texts, lang='en'):
    filtered = []
    for text in texts:
        soup = BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text()
        cleaned_text = cleaned_text.replace('\n', ' ')  # Remove newlines
        try:
            if detect(cleaned_text.strip()) == lang:
                filtered.append(cleaned_text.strip())
        except Exception as e:
            continue
    return filtered

def strip_pii(text):
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
    # Match 10-digit phone numbers with or without dashes/spaces
    text = re.sub(r'\b(?:\d{3}[-.\s]?){2}\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{12,19}\b', '[CREDIT_CARD]', text)
    return text

def remove_repetitive_ngrams(text, n=3, threshold=3):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    counts = Counter(ngrams)
    repetitive = [ngram for ngram, count in counts.items() if count >= threshold]

    for phrase in repetitive:
        # regex-safe version of the phrase
        escaped_phrase = re.escape(phrase)
        # match the phrase repeated 2+ times with optional whitespace
        print(f"******************** repetive phrase: {phrase} ********************")
        text = re.sub(rf'(?:{escaped_phrase}\s*){{{threshold},}}', phrase + ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

if __name__ == "__main__":
    # Example usage
    deduptxt = minhash_dedup(["hello world", "hello world", "goodbye world", "hello again"], threshold=0.5)
    print("Unique texts:", deduptxt )

    with open("edpypf.html", "r", encoding="utf-8") as html_file:
        html_content = html_file.read()
        cleaned_texts = clean_html_and_filter_lang([html_content])  # Pass as a list
        # print("Cleaned and filtered texts:", cleaned_texts)
    stripped_texts = [strip_pii(text) for text in cleaned_texts]
    print("Stripped PII texts:", stripped_texts[0] if stripped_texts else "No text available")

    cleaned_data = [remove_repetitive_ngrams(t) for t in stripped_texts]
    print("Final cleaned data:", cleaned_data[0] if cleaned_data else "No text available")
    
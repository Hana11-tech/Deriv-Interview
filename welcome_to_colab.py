from google.colab import drive
drive.mount('/content/drive')

BASE_PATH = "/content/drive/MyDrive/rag-system/"

import os

print(os.listdir(BASE_PATH))

import json

BASE_PATH = "/content/drive/MyDrive/rag-system/"

data = {
    "sources": [
        "https://deriv.com/help-centre/trading/",
        "https://deriv.com/help-centre/deposits-and-withdrawals/",
        "https://deriv.com/help-centre/accounts/",
        "https://deriv.com/help-centre/security/"
    ]
}

with open(BASE_PATH + "sources.json", "w") as f:
    json.dump(data, f, indent=2)

print("sources.json created at:", BASE_PATH)

import os

print(os.listdir(BASE_PATH))

import requests
from bs4 import BeautifulSoup

def scrape(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)

        if r.status_code != 200:
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        # remove obvious noise
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        # 🔥 IMPORTANT: target main content only
        main = soup.find("main")

        if main:
            text = main.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)

        return text

    except:
        return ""

all_docs = []

for url in sources:
    text = scrape(url)

    if len(text.split()) < 10:
        print("skipping (truly empty):", url)
        continue

    all_docs.append({
        "source_url": url,
        "text": text
    })

with open(BASE_PATH + "data/raw_corpus.json", "w") as f:
    json.dump(all_docs, f, indent=2)

print("saved corpus:", len(all_docs))

import json
import os

BASE_PATH = "/content/drive/MyDrive/rag-system/"

with open(BASE_PATH + "sources.json") as f:
    sources = json.load(f)["sources"]

data = []

for url in sources:
    print("scraping:", url)
    text = scrape(url)

    # If the text is empty, it means scraping failed (handled by the scrape function).
    # Skip this URL and continue to the next one.
    if not text:
        print("SKIPPING (empty content):", url)
        continue

    print("length:", len(text.split()))

    data.append({
        "source_url": url,
        "text": text
    })

out = BASE_PATH + "data/raw_corpus.json"

with open(out, "w") as f:
    json.dump(data, f, indent=2)

print("saved →", out)

with open(BASE_PATH + "data/raw_corpus.json") as f:
    data = json.load(f)

for d in data:
    print(d["source_url"])
    print("words:", len(d["text"].split()))
    print("-" * 40)

def is_valid_text(text):
    bad_signals = [
        "help centre",
        "trading ",
        "markets ",
        "payment methods",
        "tools mt5",
        "security ",
        "options "
    ]

    if len(text.split()) < 30:
        return False

    score = sum(1 for b in bad_signals if b.lower() in text.lower())

    return score < 2

import requests

url = "https://deriv.com/help-centre/accounts/"

r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
print("status:", r.status_code)
print(r.text[:1000])

import re

import json
import hashlib
import tiktoken
import os

BASE_PATH = "/content/drive/MyDrive/rag-system/"

enc = tiktoken.get_encoding("cl100k_base")

def token_count(text):
    return len(enc.encode(text))

def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()

def chunk_text(text, max_words=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = []

    for s in sentences:
        current.append(s)

        if len(" ".join(current).split()) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return [
        {
            "chunk_id": f"chunk_{i}",
            "text": c,
            "token_count": len(c.split())
        }
        for i, c in enumerate(chunks)
    ]


# load scraped data
with open(BASE_PATH + "data/raw_corpus.json") as f:
    docs = json.load(f)

all_chunks = []

for doc in docs:
    url = doc["source_url"]
    text = doc["text"]

    chunks = chunk_text(text)

    for c in chunks:
        c["source_url"] = url
        c["section_title"] = "help_article"

    all_chunks.extend(chunks)

# SAVE OUTPUT (IMPORTANT RULE)
out_path = BASE_PATH + "data/corpus.json"

with open(out_path, "w") as f:
    json.dump(all_chunks, f, indent=2)

print("chunks created:", len(all_chunks))
print("saved →", out_path)

with open(BASE_PATH + "data/raw_corpus.json") as f:
    data = json.load(f)

for i, d in enumerate(data):
    print("\nURL:", d["source_url"])
    print("TEXT LENGTH:", len(d["text"]))
    print(d["text"][:200])

with open(BASE_PATH + "data/corpus.json") as f:
    chunks = json.load(f)

print("total chunks:", len(chunks))
print(chunks[0].keys())
print(chunks[0]["text"][:200])

!pip install sentence-transformers

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_PATH = "/content/drive/MyDrive/rag-system/"

model = SentenceTransformer("all-MiniLM-L6-v2")

# load chunks
with open(BASE_PATH + "data/corpus.json") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

print("embedding chunks:", len(texts))

embeddings = model.encode(texts, show_progress_bar=True)

# save embeddings + chunks together (simple file-based vector store)
vector_store = {
    "embeddings": embeddings.tolist(),
    "chunks": chunks
}

out_path = BASE_PATH + "data/vector_store.json"

with open(out_path, "w") as f:
    json.dump(vector_store, f)

print("saved vector store →", out_path)

from sentence_transformers import SentenceTransformer
import json
import numpy as np

BASE_PATH = "/content/drive/MyDrive/rag-system/"

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(BASE_PATH + "data/corpus.json") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

embeddings = model.encode(texts, show_progress_bar=True)

store = {
    "embeddings": embeddings.tolist(),
    "chunks": chunks
}

with open(BASE_PATH + "data/vector_store.json", "w") as f:
    json.dump(store, f)

print("re-embedded:", len(chunks))

import json

with open(BASE_PATH + "data/vector_store.json") as f:
    store = json.load(f)

print("vectors:", len(store["embeddings"]))
print("chunks:", len(store["chunks"]))

import json
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_PATH = "/content/drive/MyDrive/rag-system/"

model = SentenceTransformer("all-MiniLM-L6-v2")

# load vector store
with open(BASE_PATH + "data/vector_store.json") as f:
    store = json.load(f)

embeddings = np.array(store["embeddings"])
chunks = store["chunks"]

def cosine(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def hybrid_score(query, chunk_text, base_score):
    query_terms = set(query.lower().split())
    chunk_terms = set(chunk_text.lower().split())

    overlap = len(query_terms & chunk_terms)

    return base_score + (0.05 * overlap)


def retrieve(query):
    q_emb = model.encode(query)

    scores = []

    for i, emb in enumerate(embeddings):
        base = cosine(q_emb, emb)

        # 🔥 ADD THIS BOOST
        boosted = hybrid_score(query, chunks[i]["text"], base)

        scores.append((boosted, i))

    scores.sort(reverse=True, key=lambda x: x[0])

    top = scores[:5]

    results = []
    for score, i in top:
        results.append({
            "score": float(score),
            "chunk": chunks[i]
        })

    return results

def answer_query(query):
    results = retrieve(query)

    if not results:
        return {
            "answer": "No relevant content found in Help Centre.",
            "fallback": True,
            "top_source": None,
            "score": 0.0
        }

    top_score = max(r["score"] for r in results)

    print("top score:", top_score)

    # 🔥 FIXED THRESHOLD
    if top_score < 0.70:
        return {
            "answer": "I couldn't confidently find this in the help centre content.",
            "fallback": True,
            "top_source": results[0]["chunk"]["source_url"],
            "score": top_score
        }

    context = "\n\n".join([r["chunk"]["text"] for r in results[:2]])

    answer = f"""
Based on Deriv Help Centre content:

{context[:1200]}

Sources: {[r['chunk']['chunk_id'] for r in results]}
"""

    return {
        "answer": answer,
        "fallback": False,
        "score": top_score,
        "sources": [r["chunk"]["chunk_id"] for r in results]
    }

def clean_chunk(text):
    bad_words = ["learn more", "partners", "languages", "login", "open account", "trader's hub"]
    score = sum(1 for w in bad_words if w.lower() in text.lower())
    return score < 2

context = "\n\n".join(
    r["chunk"]["text"]
    for r in results[:3]
    if clean_chunk(r["chunk"]["text"])
)

print(answer_query("How do I reset my 2FA if I lost my authenticator app?"))

print(answer_query("How do I reset 2FA?"))

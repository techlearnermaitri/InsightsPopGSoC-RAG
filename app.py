import gradio as gr
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------
# Fetch transcripts
# -------------------
BASE_URL = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds006067/main"
FILES = [
    "sub-001/func/sub-001_task-thinkaloud_transcript.txt",
    "sub-002/func/sub-002_task-thinkaloud_transcript.txt"
]

def fetch_transcripts():
    texts = []
    for f in FILES:
        url = f"{BASE_URL}/{f}"
        r = requests.get(url)
        if r.status_code == 200:
            texts.append(r.text)
    return texts

transcripts = fetch_transcripts()

# -------------------
# Preprocess: split into sentences
# -------------------
all_sentences = []
for t in transcripts:
    all_sentences.extend([s.strip() for s in t.split('.') if s])

# -------------------
# Embeddings
# -------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_sentences, convert_to_numpy=True)

# -------------------
# FAISS index
# -------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -------------------
# Chatbot function
# -------------------
def respond(message, history):
    q_vec = model.encode([message], convert_to_numpy=True)
    D, I = index.search(q_vec, k=3)  # top 3 similar sentences
    context = "\n".join([all_sentences[i] for i in I[0]])
    reply = f"🧠 Relevant transcript context:\n{context}"
    return history + [(message, reply)]

# -------------------
# Launch Gradio
# -------------------
gr.ChatInterface(respond).launch(server_name="0.0.0.0", server_port=10000)

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import chromadb
from chromadb.config import Settings
import ollama
import fitz  


# embeds using ollama, returns one vector per string
def embedTexts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = ollama.embed(model="nomic-embed-text", input=texts)
    return resp["embeddings"]

# embeds in batches to help
def embeddingBatches(texts: List[str], batchSize: int = 64) -> List[List[float]]:
    allVectors: List[List[float]] = []
    for i in range(0, len(texts), batchSize):
        batch = texts[i:i+batchSize]
        resp = ollama.embed(model="nomic-embed-text", input=batch)
        allVectors.extend(resp["embeddings"])
    return allVectors

# adds to chroma in batches to help
def chromaBatches(coll, ids, metadatas, documents, embeddings, batchSize: int = 256):
    for i in range(0, len(documents), batchSize):
        coll.add(
            ids=ids[i:i+batchSize],
            metadatas=metadatas[i:i+batchSize],
            documents=documents[i:i+batchSize],
            embeddings=embeddings[i:i+batchSize],
        )

# splits the chunk based on text
def chunkText(text: str, chunkSize: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    i = 0
    step = max(1, chunkSize - overlap)
    while i < len(text):
        chunk = text[i:i+chunkSize]
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks

# gets the plaintext from the pdfs 
def getText(pdfPath: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with fitz.open(pdfPath) as doc:
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text") or ""
            if txt.strip():
                out.append((i, txt))
    return out

# create the vector index from the pdfs
def indexDocumentText(chromaClient, collectionName="battery_corpus") -> int:

    # redo collection
    try:
        chromaClient.delete_collection(name=collectionName)
    except Exception:
        pass
    coll = chromaClient.get_or_create_collection(name=collectionName)

    pdfPaths = sorted(set(
        list(Path("./corpus/papers").rglob("*.pdf")) + list(Path("./corpus/papers").rglob("*.PDF"))
    ))

    textDocument: List[str] = []
    textMetadata: List[Dict[str, Any]] = []
    textID:   List[str] = []

    # make the chunks from the pdfs
    for pdfPath in pdfPaths:
        pages = getText(str(pdfPath))
        for (page_no, page_text) in pages:
            for j, chunk in enumerate(chunkText(page_text)):
                uid = f"pdf::{pdfPath.name}::p{page_no}::c{j}"
                textDocument.append(chunk)
                textMetadata.append({
                    "source_type": "pdf",
                    "source": pdfPath.name,
                    "page": page_no,
                    "chunk_id": j,
                })
                textID.append(uid)

    if not textDocument:
        return 0

    vectors = embeddingBatches(textDocument, batchSize=64)
    chromaBatches(coll, textID, textMetadata, textDocument, vectors, batchSize=256)

    return len(textID)

def query(chromaClient, question: str, collectionName="battery_corpus",
          amtKDocs: int = 5, **kwargs) -> Dict[str, Any]:
   
    if "amtDocs" in kwargs and isinstance(kwargs["amtDocs"], int):
        amtKDocs = kwargs["amtDocs"]

    coll = chromaClient.get_or_create_collection(name=collectionName)
    qvec = embedTexts([question])[0]
    res = coll.query(query_embeddings=[qvec], n_results=amtKDocs)
    out = []
    if res and res.get("documents"):
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            out.append({"text": doc, "meta": meta})
    return {"matches": out}

def createPrompt(
    question: str,
    retrieved: List[Dict[str, Any]],
) -> str:
    seen = set()
    contextFromDocs = []

    for item in retrieved:
        meta = item["meta"]
        text = item["text"].strip()

        if meta.get("source_type") == "pdf":
            key = (meta.get("source"), meta.get("page"))
            if key in seen:
                continue
            seen.add(key)
            citations = f"[PDF: {meta.get('source')} p.{meta.get('page')}]"
            contextFromDocs.append(f"{citations}\n{text}")

    context = "\n\n---\n\n".join(contextFromDocs) if contextFromDocs else "No documents were used"

    prompt = f"""You are a helpful battery materials research assistant
    Answer the questions using ONLY the provided context, and make up nothing
    Avoid starting the reply with the words "Based on the provided context".
    If there is not enough context to anwser, say so in few words and suggest what data would help
    Return clear conclusions and include the citation tags after, for example: [PDF: file.pdf p.3]

Question:
{question}

Context:
{context}

Rules:
- Do not make up anything that is not in the conext
- Give bullent points when appropiate 
- Do not start replies with the words "Based on the provided context"
- Keep responses under 300 words, unless the user asks for longer responses
"""
    return prompt

def createAnwser(prompt: str, model: str = "llama2") -> str:
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.2}
    )
    return resp.get("response", "").strip()

# streamlit UI

st.set_page_config(page_title="Battery Chatbot")

st.title("Battery Chatbot")
st.caption("Using Embedding model nomic-embed-text by Ollama, and using Chat model llama2 by Ollama")

# rebuild index
if st.button("Rebuild Index", key="rebuild_index"):
    idx_client = chromadb.PersistentClient(
        path="./batterydb",
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )
    with st.spinner("Rebuilding index from PDFs..."):
        n_chunks = indexDocumentText(idx_client)
    st.success(f"Indexed {n_chunks} text chunks.")

# does the asking and getting of anwsers
question = st.text_area(
    "Ask a question about battery materials",
    height=120,
    placeholder="Enter your question here"
)
ask = st.button("Get Answer", key="search_answer")

if ask and question.strip():
    client = chromadb.PersistentClient(
        path="./batterydb",
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )

    with st.spinner("Getting the context"):
        res = query(client, question, amtDocs=5)
        retrieved = res["matches"]

    if not retrieved:
        st.info("Not enough information in the current documents")
    else:
        prompt = createPrompt(question, retrieved)

        with st.spinner("Generating answer"):
            answer = createAnwser(prompt, model="llama2")

        st.subheader("Answer")
        st.write(answer)

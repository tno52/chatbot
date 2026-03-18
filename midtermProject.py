import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import streamlit as st
import chromadb
from chromadb.config import Settings
import ollama
import fitz  
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


##TESTING CHANGES##
#  HELPERS COPIED OVER FROM HW1 CHATBOT


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

def get_docs(retrieved: List[Dict[str, Any]]) -> str:
    seen = set()
    blocks: List[str] = []

    for item in retrieved:
        meta = item.get("meta", {})
        text = (item.get("text") or "").strip()
        if not text:
            continue

        if meta.get("source_type") == "pdf":
            key = (meta.get("source"), meta.get("page"))
            if key in seen:
                continue
            seen.add(key)
            tag = f"[PDF: {meta.get('source')} p.{meta.get('page')}]"
            blocks.append(f"{tag}\n{text}")

    return "\n\n---\n\n".join(blocks) if blocks else "No documents were used"


#########PROMPT ENGINEEERING###############################

llm = ChatOllama(
    model="huggingface.co/unsloth/Qwen3-0.6B-GGUF:latest",
    validate_model_on_init=True,
    num_predict=512,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    think=False
)

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

examples = [
    {"input": "How does temperature affect lithium-ion battery performance?", 
     "output": "At low temperatures, ion mobility decreases, causing higher internal resistance and reduced capacity. At high temperatures, side reactions accelerate, leading to faster degradation and potential thermal runaway."},
    {"input": "Why are batteries worse in cold weather?", 
     "output": "Low temperatures slow lithium-ion movement and increase internal resistance."},
    {"input": "Why are solid-state batteries safer?", 
     "output": "They use solid electrolytes that cannot leak or catch fire like liquid ones."},
    {"input": "What causes a battery to expand?", 
     "output": "Gas forms from reactions inside the cell during overcharging or aging."},
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a helpful battery materials research assistant. Keep your explainations short, and simplistic"
    "Do not include any informaation that is not in the provided context, unless it is "
    "in previous messages"
    "Do not make up any information. If there is not enough context, state that there is not enough context"
    "Keep your responses to under 300 words, unless the user asks for a longer explaination"
    "If the user asks to recall previous question, respond using the most recent HumanMessage from the conversation history"
    "if there is no priror human message, say: You have yet to say anything"
)

human = HumanMessagePromptTemplate.from_template(
    """Using ONLY the provided context below, unless it refers to a previous message, answer the question.
If the context is not enough, state clearly. Please do not make up any information, and do not start your
resposne with Based on the provided context. 

Context:
{context}

Question:
{question}


"""
)

chat_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    few_shot_prompt,
    human,
])

prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        few_shot_prompt,
        human,
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ]
)

chain = prompt | llm 

store = {}

#streamlit was not storing the chat history properly
if "lc_store" not in st.session_state:
    st.session_state["lc_store"] = {}

def get_session_history(session_id: str):
    # to help with streamlit issue
    store = st.session_state["lc_store"]
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap chain with message history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    history_messages_key="history",   # matches MessagesPlaceholder
    input_messages_key="input",       # matches MessagesPlaceholder
)

# Simulate conversation session
session_id = "user123"

# More concrete implementation
def summarize_history_if_long(history, llm, summary_message=None, max_len=20):
    """If the history is long, summarize it into one message."""
    print(f"History length: {len(history)}")
    if len(history) > max_len:
        summary_prompt = (
            "Summarize the following conversation so far into one concise message.\n"
            "Focus on key facts, roles, and context.\n\n"
            f"{[m.content for m in history]}"
        )
        summary_message = llm.invoke([HumanMessage(content=summary_prompt)])
        # Return a new short history that keeps only summary
        print("HISTORY:", history)
        return [AIMessage(content=summary_message.content)]
    
    return history

chain = (
    RunnablePassthrough.assign(
        history=lambda x: summarize_history_if_long(x["history"], llm)
    )
    | prompt
    | llm
)

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    history_messages_key="history",
    input_messages_key="input",
)





def create_answer(question: str, retrieved: List[Dict[str, Any]], session_id: str) -> str:
    context_str = get_docs(retrieved)
    out = chain_with_history.invoke(
        {
            "question": question,
            "context": context_str,
            "input": [HumanMessage(content=question)],
        },
        config={"configurable": {"session_id": session_id}},
    )
    return getattr(out, "content", str(out))


# streamlit UI

st.set_page_config(page_title="Battery Chatbot")
st.title("Battery Chatbot")
st.caption("Using Embedding model nomic-embed-text by Ollama, and using Chat model llama2 by Ollama")

# rebuild index
cols = st.columns([1, 1, 2])
with cols[0]:
    if st.button("Rebuild Index", key="rebuild_index"):
        idx_client = chromadb.PersistentClient(
            path="./batterydb",
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )
        with st.spinner("Rebuilding index from PDFs"):
            n_chunks = indexDocumentText(idx_client)
        st.success(f"Indexed {n_chunks} text chunks.")

with cols[1]:
    st.write("") 

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "user123"
session_id = st.session_state["session_id"] 

# does the asking and getting of anwsers
question = st.text_area(
    "Ask a question about battery materials",
    height=120,
    placeholder="Enter your question here",
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
        with st.spinner("Generating answer"):
            try:
                result = create_answer(question, retrieved, session_id=session_id)
                st.subheader("Answer:")
                st.write(result)
            except Exception as e:
                st.error(f"Generation failed: {e}")


import gc
import os
import shutil
import time
import re
from pathlib import Path
import streamlit as st
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import urllib.request
import json

DB_PATH = "chroma_store"
DOCS_PATH = "api_docs"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"

def check_ollama_processor():
    try:
        req = urllib.request.urlopen("http://localhost:11434/api/ps")
        data = json.loads(req.read().decode())
        models = data.get("models", [])
        if not models:
            return "Ollama Running (No models loaded)"
        res_info = []
        for m in models:
            name = m.get("name", "unknown")
            size_vram = m.get("size_vram", 0)
            processor = "GPU" if size_vram > 0 else "CPU"
            res_info.append(f"{name}: {processor}")
        return ", ".join(res_info)
    except Exception as e:
        return f"Unknown (Error checking: {e})"


@st.cache_resource
def get_llm(model_name, temperature=0, num_predict=512):
    # keep_alive='30m' prevents Ollama from evicting the model from RAM
    # between queries, eliminating the reload overhead on every 3rd+ question.
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict,
        keep_alive="30m"
    )

@st.cache_resource
def get_embeddings(model_name):
    return OllamaEmbeddings(model=model_name)

@st.cache_resource
def get_vectorstore(db_path, collection_name, model_name):
    embeddings = get_embeddings(model_name)
    client = chromadb.PersistentClient(path=db_path)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )

@st.cache_resource
def get_retriever(db_path, collection_name, model_name, k=3):
    vectorstore = get_vectorstore(db_path, collection_name, model_name)
    return vectorstore.as_retriever(search_kwargs={"k": k})

@st.cache_data
def load_docs(path):
    docs = []
    p = Path(path)
    if p.exists():
        for f in sorted(p.glob("*.md")):
            try:
                loader = TextLoader(str(f), encoding="utf-8")
                loaded = loader.load()
                for d in loaded:
                    d.metadata["source_file"] = f.name
                docs.extend(loaded)
            except Exception as e:
                print(f"Error loading {f.name}: {e}")
    return docs

def search_documents(term):
    matches = []
    docs = load_docs(DOCS_PATH)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    for i, s in enumerate(splits):
        s.metadata["chunk_id"] = i
        if term.lower() in s.page_content.lower():
            matches.append((s, i))
    return matches

def verify_chunks_and_metadata(splits):
    counts = {}
    for s in splits:
        src = s.metadata.get("source_file", "unknown")
        counts[src] = counts.get(src, 0) + 1
        
    print("Chunks indexed per file:")
    for file, cnt in counts.items():
        print(f"  {file} -> {cnt} chunks")
        
    print("Chunk metadata verification:")
    for i, s in enumerate(splits):
        src = s.metadata.get("source_file", "unknown")
        cid = s.metadata.get("chunk_id", "n/a")
        length = len(s.page_content)
        print(f"  Chunk {i}: Source={src}, Chunk ID={cid}, Content Length={length}")

@st.cache_data
def is_query_related(query):
    classification_template = """You are a classification assistant. Determine if the user's question is related to the API documentation or API reference topics.
The documentation only covers the following topics:
- API Authentication, generating/revoking API keys, and authorization
- Rate limiting, status codes (like 429), and exponential backoff
- Pagination, cursors, limits, and page parameters
- Webhooks, retries, and notifications
- API versioning, v1 deprecation, and migrating to v2
- API endpoints and error handling formats

If the query is related to any of these topics (including general questions about the API), respond with exactly 'RELATED'.
If the query is unrelated (e.g. weather, sports/cricket, movie recommendations, politics, general chat, unrelated code), respond with exactly 'UNRELATED'.

Query: {query}
Response:"""
    
    prompt = PromptTemplate.from_template(classification_template)
    prompt_val = prompt.format(query=query)
    
    llm = get_llm(model_name=LLM_MODEL, temperature=0, num_predict=16)
    try:
        res = llm.invoke(prompt_val)
        classification = res.content.strip().upper()
        print("Guardrail Classification:", classification)
        if "UNRELATED" in classification:
            return False
        return "RELATED" in classification
    except Exception as e:
        print("Guardrail classification failed:", e)
        return True

def rephrase_query_with_history(query, history):
    # Removed the extra LLM rephrasing call — it added a full ~10-15s penalty
    # on every question after the 1st. The chat history is already injected
    # into the main answer prompt's history_text block, so the model handles
    # follow-up context automatically without a separate call.
    return query



def rebuild_index():
    # Progress placeholders
    status = st.empty()
    
    # Clear caches
    load_docs.clear()
    get_vectorstore.clear()
    get_retriever.clear()
    
    # 1. Close any active Chroma connections
    status.write("Closing active database connections...")
    if "vectorstore" in st.session_state:
        vs = st.session_state["vectorstore"]
        if hasattr(vs, "_client") and vs._client is not None:
            try:
                vs._client.close()
            except Exception:
                pass
        del st.session_state["vectorstore"]
    if "chroma_client" in st.session_state:
        try:
            st.session_state["chroma_client"].close()
        except Exception:
            pass
        del st.session_state["chroma_client"]
    
    gc.collect()
    time.sleep(0.5)  # Allow file locks to release on Windows
    
    # 2. Delete corrupted Chroma store safely
    status.write("Deleting old store...")
    db_dir = Path(DB_PATH)
    if db_dir.exists():
        try:
            shutil.rmtree(db_dir)
        except Exception as e:
            st.error(f"Error removing old store directory: {e}")
            return
            
    # 3. Reload documents
    status.write("Loading docs...")
    docs = load_docs(DOCS_PATH)
    if not docs:
        st.error("No markdown documents found in api_docs/")
        return
        
    # 4. Chunk documents
    status.write("Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    for i, s in enumerate(splits):
        s.metadata["chunk_id"] = i
        
    verify_chunks_and_metadata(splits)
        
    # 5. Generate embeddings & 6. Rebuild Chroma collection
    status.write("Generating embeddings & building vector store...")
    vectorstore = get_vectorstore(DB_PATH, "api_docs_collection", EMBEDDING_MODEL)
    vectorstore.add_documents(splits)
    st.session_state["vectorstore"] = vectorstore
    
    # 7. Reload retriever
    status.write("Retriever ready...")
    retriever = get_retriever(DB_PATH, "api_docs_collection", EMBEDDING_MODEL, k=3)
    st.session_state["retriever"] = retriever
    
    # Validation checks
    count = vectorstore._collection.count()
    if count == 0:
        status.error("Index build failed: Vector count is 0.")
        return
        
    # 8. Verify retrieval automatically
    status.write("Completed.")
    test_query = "How does authentication work?"
    test_docs = retriever.invoke(test_query)
    
    if len(test_docs) == 0:
        status.error("Index build failed: Retrieval query verification returned 0 chunks.")
        return
        
    # Set session state flags
    st.session_state["index_built"] = True
    st.session_state["files_loaded"] = len(list(set(d.metadata["source_file"] for d in docs)))
    st.session_state["chunks_created"] = len(splits)
    st.session_state["vector_count"] = count
    
    st.success("Index rebuild completed successfully!")

from langchain_core.prompts import PromptTemplate
import re

def ask_question(query):
    if "retriever" not in st.session_state or st.session_state["retriever"] is None:
        st.error("Index not initialized. Please rebuild index first.")
        return
        
    retriever = st.session_state["retriever"]
    vectorstore = st.session_state.get("vectorstore")
    
    # Input Guardrail Check
    if not is_query_related(query):
        st.warning("This assistant only answers questions related to the uploaded API documentation.")
        st.markdown("### Answer")
        st.write("This assistant only answers questions related to the uploaded API documentation.")
        return
        
    # Rephrase follow-up questions if history exists
    history = st.session_state.get("chat_history", [])
    search_query = rephrase_query_with_history(query, history)
    
    # 4. Search Utility: check if significant query terms exist in raw documents
    keywords = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in search_query.split()]
    keywords = [w for w in keywords if len(w) > 4] # ignore short words
    if keywords:
        docs_raw = load_docs(DOCS_PATH)
        has_any = False
        for d in docs_raw:
            content_lower = d.page_content.lower()
            if any(kw.lower() in content_lower for kw in keywords):
                has_any = True
                break
        if not has_any:
            # 5. If the term does not exist in indexed docs, show exact message
            st.warning("Topic not present in uploaded documentation.")
            st.markdown("### Answer")
            st.write("Topic not present in uploaded documentation.")
            return

    # 1. Document retrieval and timing (Retrieve 3 to optimize performance)
    retrieval_start = time.perf_counter()
    if vectorstore is not None:
        docs_with_score = vectorstore.similarity_search_with_score(search_query, k=3)
    else:
        # Fallback to standard retriever invoke
        docs_raw = retriever.invoke(search_query)
        docs_with_score = [(d, 0.0) for d in docs_raw][:3]
        
    retrieval_time = time.perf_counter() - retrieval_start
    
    # 1. Add debug output showing Retrieved documents: filename, chunk id, similarity score
    print("Retrieved documents:")
    for i, (doc, score) in enumerate(docs_with_score):
        src = doc.metadata.get("source_file", "unknown")
        cid = doc.metadata.get("chunk_id", "n/a")
        print(f"  - Filename: {src}, Chunk ID: {cid}, Similarity Score: {score}")
        
    # Print top retrieval scores
    print("Top retrieval scores:")
    for i, (_, score) in enumerate(docs_with_score):
        print(f"  Score {i+1}: {score}")
        
    # Limit to top 3 for prompt context to improve performance
    docs = [d for d, _ in docs_with_score][:3]
    
    llm = get_llm(model_name=LLM_MODEL, temperature=0, num_predict=256)
    
    # Fallback logic should execute ONLY when len(docs) == 0
    if len(docs) == 0:
        st.warning("No relevant chunks found in the documentation.")
        st.markdown("### Answer")
        st.write("Information not found in the uploaded documentation.")
        return
        
    # 2. Prompt creation and timing
    prompt_start = time.perf_counter()
    # Format context
    context = "\n\n".join(d.page_content for d in docs)
    print(f"\n--- Context sent to LLM ({len(docs)} chunks) ---\n{context}\n---\n")
    
    # Format history text
    history_text = ""
    if history:
        for u, a in history[-5:]:
            history_text += f"User: {u}\nAssistant: {a}\n\n"
    else:
        history_text = "(None)"
        
    # Exact strict context grounding system prompt
    template = """You are an API documentation assistant.

Use the retrieved context below to answer the user's question.
The context contains API documentation — understand what the user is asking about and find the relevant information in the context.
For example: if the user asks about 'header', look for Authorization headers, Accept headers, or any HTTP headers mentioned in the context.

Rules:
- Answer based on information present in the context.
- You may connect related concepts (e.g. 'header' refers to Authorization, Accept headers etc.).
- Be concise and direct.
- If the context truly contains no relevant information at all, respond with exactly:
  "Information not found in the uploaded documentation."

Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    prompt_val = prompt.format(history_text=history_text, context=context, question=query)
    prompt_time = time.perf_counter() - prompt_start
    
    # 3. LLM Generation and timing
    gen_start = time.perf_counter()
    model_load_time = 0.0
    processor_info = "Unknown"
    
    st.markdown("### Answer")
    answer_placeholder = st.empty()
    answer = ""
    
    with st.spinner("Thinking..."):
        try:
            for chunk in llm.stream(prompt_val):
                answer += chunk.content
                answer_placeholder.markdown(answer + "▌")
            answer_placeholder.markdown(answer)
            processor_info = check_ollama_processor()
        except Exception as e:
            answer = f"Error during generation: {e}"
            answer_placeholder.write(answer)
            
    gen_time = time.perf_counter() - gen_start
    total_time = retrieval_time + prompt_time + gen_time
    
    # Performance logging
    print("----- Performance Audit -----")
    print(f"Retrieval Time: {retrieval_time:.4f} s")
    print(f"Prompt Construction Time: {prompt_time:.4f} s")
    print(f"LLM Generation Time: {gen_time:.4f} s")
    print(f"Ollama Processor: {processor_info}")
    print(f"Total Request Time: {total_time:.4f} s")
    print("-----------------------------")
    
    # Empty answer check
    if not answer or answer.strip() == "":
        st.error("LLM returned empty response.")
        print("ERROR: LLM returned empty response.")
        answer = "Error: The model returned an empty response. Please verify Ollama configuration."
    
    # Handle missing information response
    if "information not found in the uploaded documentation" in answer.lower():
        cleaned_answer = "Information not found in the uploaded documentation."
    else:
        # Post-processing: Remove sentences containing blocked disclaimer phrases
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        filtered_sentences = []
        blocked_phrases = [
            "information not found",
            "not enough context",
            "context unavailable",
            "beyond this explanation",
            "unable to determine"
        ]
        for s in sentences:
            s_lower = s.lower()
            if any(p in s_lower for p in blocked_phrases):
                continue
            filtered_sentences.append(s)
        
        cleaned_answer = " ".join(filtered_sentences).strip()
        if cleaned_answer == "":
            cleaned_answer = answer
            
    # Update placeholder with final cleaned answer
    answer_placeholder.markdown(cleaned_answer)
    
    # Append to memory limit to 5
    st.session_state["chat_history"].append((query, cleaned_answer))
    if len(st.session_state["chat_history"]) > 5:
        st.session_state["chat_history"] = st.session_state["chat_history"][-5:]
        
    # Timing in UI
    st.markdown(f"**Timing** — Retrieval: {retrieval_time:.2f}s | Prompt: {prompt_time:.2f}s | Generation: {gen_time:.2f}s | Total: {total_time:.2f}s")
    st.markdown(f"**Processor/Hardware**: {processor_info}")
    
    st.markdown("### Sources")
    seen_sources = set()
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            st.markdown(f"- {src}")

# Page setup
st.title("📘 API Documentation Assistant")

# Startup Initialization
startup_start = time.perf_counter()
if "index_built" not in st.session_state:
    st.session_state["index_built"] = False

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

db_dir = Path(DB_PATH)
db_exists = db_dir.exists() and (db_dir / "chroma.sqlite3").exists()

if not db_exists:
    if not st.session_state["index_built"]:
        rebuild_index()
else:
    if not st.session_state["index_built"]:
        try:
            vectorstore = get_vectorstore(DB_PATH, "api_docs_collection", EMBEDDING_MODEL)
            retriever = get_retriever(DB_PATH, "api_docs_collection", EMBEDDING_MODEL, k=3)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["retriever"] = retriever
            st.session_state["index_built"] = True
            
            count = vectorstore._collection.count()
            st.session_state["files_loaded"] = len(list(Path(DOCS_PATH).glob("*.md")))
            st.session_state["chunks_created"] = count
            st.session_state["vector_count"] = count
        except Exception as e:
            print(f"Error loading index on startup: {e}")

startup_time = time.perf_counter() - startup_start
if "startup_time" not in st.session_state or startup_time > 0.01:
    st.session_state["startup_time"] = startup_time

# Sidebar for Rebuild Index
with st.sidebar:
    st.header("Index Management")
    if st.button("Initialize / Rebuild Index"):
        rebuild_index()
        
    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared!")

    # Performance / Timing Display
    st.markdown("---")
    st.markdown(f"**Startup Time**: {st.session_state.get('startup_time', 0.0):.2f}s")
    
    # Validation Display
    if st.session_state.get("index_built", False):
        st.markdown(f"**Files Loaded**: {st.session_state.get('files_loaded', 0)}")
        st.markdown(f"**Chunks Created**: {st.session_state.get('chunks_created', 0)}")
        st.markdown(f"**Vector Count**: {st.session_state.get('vector_count', 0)}")

# Display chat history above the input box
if st.session_state.get("chat_history"):
    st.header("Chat History")
    for u, a in st.session_state["chat_history"]:
        st.markdown(f"**User**: {u}")
        st.markdown(f"**Assistant**: {a}")
        st.markdown("---")

# Main Question Answering area
st.header("Ask Question")
query = st.text_input("Enter your query about the API documentation:")
if st.button("Ask"):
    if query:
        ask_question(query)

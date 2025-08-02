# agentic_rag_local.py  – Agentic‑RAG demo with streaming and diagnostics
# Licence: MIT   (c) 2024‑2025  LangChain AI & contributors

import os, tempfile, time
from pathlib import Path
from typing import List, Iterator
from datetime import datetime

import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    WebBaseLoader, PyPDFLoader, UnstructuredFileLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from langgraph.graph import MessagesState, StateGraph, START, END

# ────────────────────────── 0.  Environment ────────────────────────── #
st.set_page_config("Agentic‑RAG (local)", "🤖", layout="wide")
load_dotenv()                              # optional; does nothing if .env absent

# Check Ollama connectivity
@st.cache_data(ttl=60)  # Cache for 1 minute
def check_ollama_status():
    """Check if Ollama is running and which models are available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, [m["name"] for m in models]
        return False, []
    except:
        return False, []

# ─────────────────── 1.  Local models via Ollama ───────────────────── #
# Embeddings - only use supported parameters
embeddings = OllamaEmbeddings(
    model="all-minilm:latest",              # pull once:  ollama pull nomic-embed-text
    base_url="http://localhost:11434"      # default Ollama REST endpoint
)

# Chat / reasoning model with streaming support
llm = ChatOllama(
    model="qwen2.5:0.5b",                      # pull once:  ollama pull qwen2.5:0.5b
    base_url="http://localhost:11434",
    temperature=0.2,                       # lower temp for more consistent output
    num_ctx=8000,                          # Gemma 2B supports 8k context
    num_batch=96,                          # safe batch size for 32 GB RAM
    num_thread=os.cpu_count() or 8,
    streaming=True                         # Enable streaming
)

response_model = query_reformulation_model = intent_model = llm

# ─────────────────────── 2.  Session state init ────────────────────── #
for k, v in {
    "graph": None,
    "chat_history": [],
    "message_history": [],
    "urls_loaded": [],
    "files_loaded": [],
    "performance_metrics": {},
    "diagnostics": []
}.items():
    st.session_state.setdefault(k, v)

# ──────────────────────── 3.  Helper functions ─────────────────────── #

def time_it(func_name):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            if func_name not in st.session_state.performance_metrics:
                st.session_state.performance_metrics[func_name] = []
            st.session_state.performance_metrics[func_name].append(elapsed)
            return result
        return wrapper
    return decorator

def add_diagnostic(step: str, detail: str = "", elapsed: float = None):
    """Add a diagnostic entry"""
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "step": step,
        "detail": detail,
        "elapsed": elapsed
    }
    st.session_state.diagnostics.append(entry)

@time_it("HTML Cleaning")
def clean_html_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()
    for a in soup.find_all("a"):
        if not a.get_text(strip=True):
            a.decompose()
    return "\n".join(
        ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()
    )

@time_it("URL Loading")
def docs_from_urls(urls: List[str]):
    out = []
    for u in urls:
        try:
            for d in WebBaseLoader(u).load():
                d.page_content = clean_html_content(d.page_content)
                out.append(d)
        except Exception as e:
            st.warning(f"Could not load {u}: {e}")
    return out

@time_it("File Loading")
def docs_from_uploaded(files):
    out: List = []
    for up in files or []:
        suffix = Path(up.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up.read()); tmp_path = tmp.name
        try:
            if suffix == ".pdf":
                first = PyPDFLoader(tmp_path).load()
                out.extend(first if any(d.page_content.strip() for d in first)
                           else UnstructuredFileLoader(tmp_path, mode="elements").load())
            elif suffix in {".docx", ".doc"}:
                out.extend(UnstructuredFileLoader(tmp_path, mode="elements").load())
            elif suffix == ".txt":
                out.extend(TextLoader(tmp_path, autodetect_encoding=True).load())
            else:
                st.warning(f"Unsupported file type: {up.name}")
        except Exception as e:
            st.warning(f"Could not read {up.name}: {e}")
    return out

@time_it("Document Splitting")
def split_documents(docs):
    # Dynamic chunk size based on total content
    total_chars = sum(len(d.page_content) for d in docs)
    
    # Adjust chunk size based on document size
    if total_chars < 10000:  # Small docs
        chunk_size = 200
        chunk_overlap = 20
    elif total_chars < 50000:  # Medium docs
        chunk_size = 300
        chunk_overlap = 50
    else:  # Large docs
        chunk_size = 400
        chunk_overlap = 80
    
    st.info(f"📏 Using chunk size: {chunk_size} (based on {total_chars:,} total characters)")
    
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ).split_documents(docs)

@time_it("Building Retriever")
def build_retriever_tool(splits):
    # Add manual progress tracking for embeddings
    progress_container = st.container()
    progress_bar = progress_container.progress(0.0)
    status_text = progress_container.empty()
    
    # Process in batches for better performance and monitoring
    batch_size = 10  # Process 10 chunks at a time
    total_chunks = len(splits)
    
    # Time individual embedding calls
    embedding_times = []
    
    # Initialize vector store with first batch
    batch_start = time.time()
    status_text.text(f"Embedding chunks 1-{min(batch_size, total_chunks)} of {total_chunks}...")
    
    # Create vector store with timing
    start_embed = time.time()
    vect = InMemoryVectorStore.from_documents(
        splits[:batch_size], 
        embedding=embeddings
    )
    first_batch_time = time.time() - start_embed
    embedding_times.append(first_batch_time)
    
    progress_bar.progress(min(batch_size, total_chunks) / total_chunks)
    status_text.text(f"First batch took {first_batch_time:.2f}s ({first_batch_time/min(batch_size, total_chunks):.2f}s per chunk)")
    
    # Add remaining chunks in batches
    for i in range(batch_size, total_chunks, batch_size):
        batch_start = time.time()
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = batch_end - i
        
        status_text.text(f"Embedding chunks {i+1}-{batch_end} of {total_chunks}...")
        vect.add_documents(splits[i:batch_end])
        
        batch_time = time.time() - batch_start
        embedding_times.append(batch_time)
        avg_time_per_chunk = sum(embedding_times) / ((i + batch_chunks) / batch_size) / batch_size
        
        progress_bar.progress(batch_end / total_chunks)
        status_text.text(f"Batch took {batch_time:.2f}s. Avg: {avg_time_per_chunk:.2f}s/chunk. ETA: {avg_time_per_chunk * (total_chunks - batch_end):.0f}s")
    
    # Show final stats
    total_embed_time = sum(embedding_times)
    avg_chunk_time = total_embed_time / total_chunks
    status_text.text(f"✅ Embedding complete! Total: {total_embed_time:.1f}s, Avg: {avg_chunk_time:.2f}s/chunk")
    time.sleep(2)  # Show final message briefly
    
    progress_container.empty()
    
    return create_retriever_tool(
        vect.as_retriever(search_kwargs={"k": 5}),
        name="retrieve_docs",
        description="Search the user‑supplied webpages and documents."
    )

def preview_chunks(chunks, rows=25, chars=120):
    return "\n".join(
        f"**{i:>2}**  {d.page_content.replace(chr(10), ' ')[:chars]} …"
        for i, d in enumerate(chunks[:rows], 1)
    ) or "_no text extracted_"

def show_performance_metrics():
    """Display performance metrics"""
    if st.session_state.performance_metrics:
        with st.expander("⏱️ Performance Metrics", expanded=False):
            for func_name, times in st.session_state.performance_metrics.items():
                avg_time = sum(times) / len(times)
                st.metric(
                    func_name, 
                    f"{avg_time:.2f}s avg",
                    f"Total: {sum(times):.2f}s ({len(times)} calls)"
                )

def test_embedding_speed():
    """Test embedding speed with sample texts"""
    with st.expander("🔬 Test Embedding Speed", expanded=False):
        if st.button("Run Embedding Speed Test"):
            test_sizes = [1, 10, 50]
            results = []
            
            for size in test_sizes:
                test_texts = [f"This is test text number {i} for embedding speed testing." for i in range(size)]
                start = time.time()
                try:
                    embeddings.embed_documents(test_texts)
                    elapsed = time.time() - start
                    results.append((size, elapsed, elapsed/size))
                    st.success(f"{size} embeddings: {elapsed:.2f}s ({elapsed/size:.3f}s per embedding)")
                except Exception as e:
                    st.error(f"Error testing {size} embeddings: {e}")
            
            if results:
                st.info(f"💡 Your system processes ~{1/results[-1][2]:.1f} embeddings per second")
                
        st.caption("Note: First run may be slower due to model loading")

def show_diagnostics():
    """Display diagnostic information"""
    if st.session_state.diagnostics:
        with st.expander("🔍 Diagnostics", expanded=True):
            for diag in st.session_state.diagnostics[-10:]:  # Show last 10
                elapsed_str = f" ({diag['elapsed']:.2f}s)" if diag['elapsed'] else ""
                st.text(f"{diag['timestamp']} - {diag['step']}{elapsed_str}")
                if diag['detail']:
                    st.caption(f"  → {diag['detail']}")

# ─────────────────────── 4.  Intent grading ────────────────────────── #
def classify_intent(text: str) -> str:
    try:
        resp = intent_model.invoke(
            "You are an intent classifier. "
            "If the user message below is merely an acknowledgement "
            "(e.g. ok, thanks, bye, 👍) with no new request, answer 'ACK'. "
            "Otherwise answer 'QUESTION'.\n\n"
            f"User message: {text!r}"
        )
        label = resp.content.strip().upper()
        return "ack" if label.startswith("ACK") else "question"
    except Exception:
        return "question"

# ─────────────── 5.  Query reformulation helpers ───────────────────── #
def get_conversation_summary(msgs: List[BaseMessage]) -> str:
    conv_parts = []
    for msg in msgs[-6:]:
        if isinstance(msg, HumanMessage):
            conv_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conv_parts.append(f"Assistant: {msg.content[:200]}…")
    return "\n".join(conv_parts) if conv_parts else ""

def reformulate_query_with_context(state: MessagesState) -> str:
    msgs = state["messages"]
    current_q = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
    conv_summary = get_conversation_summary(msgs[:-1])
    if not conv_summary:
        return current_q
    prompt = (
        "Given the conversation context below, reformulate the user's current question "
        "to be self‑contained for document retrieval. Include any relevant context.\n\n"
        f"Conversation context:\n{conv_summary}\n\n"
        f"Current question: {current_q}\n\n"
        "Reformulated query:"
    )
    try:
        add_diagnostic("Query Reformulation", "Analyzing conversation context")
        resp = query_reformulation_model.invoke(prompt)
        reformulated = resp.content.strip()
        add_diagnostic("Query Reformulation Complete", f"'{current_q}' → '{reformulated}'")
        return reformulated
    except Exception:
        return current_q

# ───────────────────── 6.  Retrieval & answering ───────────────────── #
def run_retrieval(state: MessagesState, tool):
    start_time = time.time()
    add_diagnostic("Retrieval Started", "Reformulating query for context")
    
    contextual_query = reformulate_query_with_context(state)
    
    try:
        add_diagnostic("Searching Documents", f"Query: '{contextual_query}'")
        ctx = tool.invoke({"query": contextual_query})
        
        elapsed = time.time() - start_time
        add_diagnostic("Retrieval Complete", f"Found {len(ctx.split('\\n\\n'))} relevant chunks", elapsed)
        
        return {"messages": [SystemMessage(
            content=f"Reformulated query: {contextual_query}\nRetrieved context:\n{ctx}"
        )]}
    except Exception as e:
        add_diagnostic("Retrieval Error", str(e))
        return {"messages": [SystemMessage(content=f"Retrieval error: {e}")]} 

def gen_answer_streaming(state: MessagesState) -> Iterator[str]:
    """Generate answer with streaming support"""
    msgs = state["messages"]
    current_q = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
    ctx, reformulated_q = "", ""
    
    for msg in reversed(msgs):
        if isinstance(msg, SystemMessage) and "Retrieved context:" in msg.content:
            reformulated_q, ctx = msg.content.split("\nRetrieved context:\n", 1)
            reformulated_q = reformulated_q.replace("Reformulated query: ", "")
            break

    conv_history = []
    for m in msgs[:-2]:
        if isinstance(m, HumanMessage):
            conv_history.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            conv_history.append(f"Assistant: {m.content}")
    conv_text = "\n".join(conv_history[-6:]) if conv_history else "This is the first question."

    prompt = f"""
You are a helpful assistant with access to a document knowledge base.

Previous conversation:
{conv_text}

Current question: {current_q}
{f'(Reformulated for search: {reformulated_q})' if reformulated_q != current_q else ''}

Retrieved context:
{ctx or 'No context available'}

Instructions:
- Answer using the retrieved context when possible.
- Reference prior conversation where relevant.
- Be concise but complete.
- If context is insufficient, say so.

Answer:"""
    
    try:
        add_diagnostic("Answer Generation Started", "Streaming response")
        stream = response_model.stream(prompt)
        full_response = ""
        
        for chunk in stream:
            if hasattr(chunk, 'content'):
                full_response += chunk.content
                yield chunk.content
        
        add_diagnostic("Answer Generation Complete", f"Generated {len(full_response)} characters")
    except Exception as e:
        add_diagnostic("Answer Generation Error", str(e))
        yield f"Error generating answer: {e}"

def gen_answer(state: MessagesState):
    """Non-streaming version for graph compatibility"""
    full_response = ""
    for chunk in gen_answer_streaming(state):
        full_response += chunk
    return {"messages": [AIMessage(content=full_response)]}

# ─────────────────────── 7.  Build LangGraph ───────────────────────── #
def compile_graph(tool):
    g = StateGraph(MessagesState)
    g.add_node("retrieve", lambda s: run_retrieval(s, tool))
    g.add_node("answer",   gen_answer)
    g.add_edge(START,       "retrieve")
    g.add_edge("retrieve",  "answer")
    g.add_edge("answer",    END)
    return g.compile()

# ─────────────────────────── 8.  Streamlit UI ───────────────────────── #
st.title("Agentic‑RAG with Conversation Memory (local)")

# Create layout
col1, col2 = st.columns([2, 1])

with col1:
    # Show performance metrics and speed test
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        show_performance_metrics()
    with col1_2:
        test_embedding_speed()

    # Performance tips
    with st.expander("💡 Performance Tips", expanded=False):
        st.markdown("""
        **If embeddings are slow:**
        1. **Use a faster model**: Try `ollama pull all-minilm:22m` (10x faster)
        2. **Reduce chunk size**: Smaller chunks = faster embeddings
        3. **Check Ollama status**: Run `curl http://localhost:11434/api/tags` in terminal
        4. **CPU vs GPU**: Ollama uses CPU by default. For GPU, check Ollama docs
        5. **Close other apps**: Free up RAM for better performance
        
        **Expected speeds (CPU):**
        - `nomic-embed-text`: ~0.1-0.5s per chunk
        - `all-minilm`: ~0.01-0.05s per chunk
        """)

with col2:
    # Show diagnostics in sidebar-like column
    show_diagnostics()

# (A) Source ingestion UI
with st.expander("Step 1 – Provide sources (URLs or files)",
                 expanded=(st.session_state.graph is None)):
    url_input = st.text_input("URLs (comma‑separated)",
                              value=",".join(st.session_state.urls_loaded))
    uploads = st.file_uploader("Upload documents (PDF, DOCX/DOC, TXT)",
                               type=["pdf", "docx", "doc", "txt"],
                               accept_multiple_files=True)
    col_a, col_b = st.columns(2)
    if col_a.button("Load and index", type="primary", use_container_width=True):
        # Clear diagnostics for new load
        st.session_state.diagnostics = []
        st.session_state.performance_metrics = {}
        
        with st.spinner("Processing…"):
            start_total = time.time()
            
            add_diagnostic("Processing Started", "Loading documents")
            urls  = [u.strip() for u in url_input.split(",") if u.strip()]
            docs  = docs_from_urls(urls) + docs_from_uploaded(uploads)
            if not docs:
                st.error("Nothing could be loaded"); st.stop()
            
            # Show document stats
            total_chars = sum(len(d.page_content) for d in docs)
            st.info(f"📊 Loaded {len(docs)} documents ({total_chars:,} characters)")
            add_diagnostic("Documents Loaded", f"{len(docs)} docs, {total_chars:,} chars")
            
            splits = split_documents(docs)
            st.info(f"✂️ Split into {len(splits)} chunks")
            add_diagnostic("Documents Split", f"{len(splits)} chunks created")
            
            # Build retriever with progress tracking
            st.info("🔄 Creating embeddings... (this may take a while)")
            tool   = build_retriever_tool(splits)
            
            st.session_state.retriever_tool = tool
            st.session_state.graph = compile_graph(tool)
            st.session_state.urls_loaded  = urls
            st.session_state.files_loaded = [up.name for up in uploads] if uploads else []
            
            total_time = time.time() - start_total
            st.success(f"✅ Indexed {len(docs)} documents into {len(splits)} chunks in {total_time:.1f}s")
            add_diagnostic("Indexing Complete", f"Total time: {total_time:.1f}s", total_time)
            
            st.markdown("##### Preview of extracted chunks")
            st.markdown(preview_chunks(splits), unsafe_allow_html=True)
            
            # Force refresh to show metrics
            st.rerun()

    if col_b.button("Reset session", use_container_width=True):
        for k in ["graph", "chat_history", "message_history",
                  "urls_loaded", "files_loaded", "performance_metrics", "diagnostics"]:
            st.session_state[k] = {} if k == "performance_metrics" else ([] if isinstance(st.session_state.get(k, []), list) else None)
        st.info("Session cleared.")

# (B) Chat interface
if st.session_state.graph:
    st.header("Step 2 – Chat with your knowledge base")

    if st.session_state.chat_history:
        st.caption(f"Conversation context: {len(st.session_state.chat_history)} turns")

    # Display chat history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content, unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask a question")
    if prompt:
        # Clear diagnostics for new query
        st.session_state.diagnostics = []
        
        add_diagnostic("User Input", prompt)
        intent = classify_intent(prompt)
        add_diagnostic("Intent Classification", f"Classified as: {intent.upper()}")
        
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        if intent == "ack":
            add_diagnostic("Generating Acknowledgment", "Quick response mode")
            ack_text = response_model.invoke(
                "Respond in ONE short, polite sentence that fits the user's tone.\n"
                f"User: {prompt}"
            ).content
            with st.chat_message("assistant"):
                st.markdown(ack_text)
            st.session_state.chat_history.append(("assistant", ack_text))
            st.session_state.message_history.extend(
                [HumanMessage(content=prompt), AIMessage(content=ack_text)]
            )
            st.session_state.message_history = st.session_state.message_history[-20:]
            add_diagnostic("Response Complete", "Acknowledgment sent")
            st.rerun()  # Refresh to show diagnostics

        # Question branch – run with streaming
        messages = st.session_state.message_history.copy()
        messages.append(HumanMessage(content=prompt))
        tmp_state = {"messages": messages.copy()}

        with st.chat_message("assistant"):
            status_container = st.empty()
            response_container = st.empty()
            
            try:
                # Step 1: Retrieval
                status_container.info("🔄 **Step 1/3:** Reformulating query and retrieving context...")
                add_diagnostic("Pipeline Started", "Beginning RAG process")
                
                ret_start = time.time()
                ret = run_retrieval(tmp_state, st.session_state.retriever_tool)
                ret_elapsed = time.time() - ret_start
                tmp_state["messages"].extend(ret["messages"])
                
                status_container.success(f"✅ **Step 1/3 Complete:** Retrieved relevant context ({ret_elapsed:.1f}s)")
                time.sleep(0.5)  # Brief pause to show status
                
                # Step 2: Answer generation with streaming
                status_container.info("✏️ **Step 2/3:** Generating answer from context...")
                
                # Stream the response
                full_response = ""
                response_start = time.time()
                
                for chunk in gen_answer_streaming(tmp_state):
                    full_response += chunk
                    # Update the response container with accumulated text
                    response_container.markdown(full_response + "▌")
                
                response_elapsed = time.time() - response_start
                
                # Remove cursor and show final response
                response_container.markdown(full_response)
                
                # Step 3: Complete
                status_container.success(f"✅ **All steps complete!** (Retrieval: {ret_elapsed:.1f}s, Generation: {response_elapsed:.1f}s)")
                time.sleep(2)  # Show completion status briefly
                status_container.empty()
                
                # Update message history
                st.session_state.message_history.extend(
                    [HumanMessage(content=prompt), AIMessage(content=full_response)]
                )
                if len(st.session_state.message_history) > 20:
                    st.session_state.message_history = st.session_state.message_history[-20:]
                
                add_diagnostic("Pipeline Complete", f"Total time: {ret_elapsed + response_elapsed:.1f}s")

            except Exception as e:
                status_container.empty()
                err = f"Error: {e}"
                st.error(err)
                add_diagnostic("Pipeline Error", str(e))
                import traceback, textwrap
                with st.expander("Traceback"):
                    st.code(textwrap.dedent(traceback.format_exc()))
                st.session_state.chat_history.append(("assistant", err))
                st.rerun()  # Refresh to show diagnostics

        st.session_state.chat_history.append(("assistant", full_response))
        st.rerun()  # Refresh to show final diagnostics
else:
    st.info("⚠️ Index at least one URL or file first.")

# (C) Footer – show indexed sources
if st.session_state.urls_loaded or st.session_state.files_loaded:
    st.caption("Sources indexed: " + ", ".join(
        st.session_state.urls_loaded + st.session_state.files_loaded))
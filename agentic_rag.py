"""
agentic_rag.py – single-file version of LangGraph’s “Agentic RAG” tutorial
Licence: MIT (copyright © 2024 LangChain AI and contributors)
"""

import os
from typing import Literal

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# 0. Environment check – requires an OpenAI API key
# ---------------------------------------------------------------------
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Set OPENAI_API_KEY in your shell before running.")

# ---------------------------------------------------------------------
# 1. Fetch and split documents
# ---------------------------------------------------------------------
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(u).load() for u in urls]
docs_flat = [d for sub in docs for d in sub]

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = splitter.split_documents(docs_flat)

# ---------------------------------------------------------------------
# 2. Vector store and retriever tool
# ---------------------------------------------------------------------
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_blog_posts",
    description="Search and return information about Lilian Weng blog posts.",
)

# ---------------------------------------------------------------------
# 3. LLM model
# ---------------------------------------------------------------------
response_model = init_chat_model("openai:gpt-4.1", temperature=0)
grader_model = init_chat_model("openai:gpt-4.1", temperature=0)

# ---------------------------------------------------------------------
# 4. Node-level functions
# ---------------------------------------------------------------------
def generate_query_or_respond(state: MessagesState):
    """Initial node: decide whether to call the retriever tool or answer directly."""
    resp = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [resp]}


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Document:\n\n{context}\n\nQuestion: {question}\n\n"
    "Reply with 'yes' if relevant, otherwise 'no'."
)


class _GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, otherwise 'no'")


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)

    result = grader_model.with_structured_output(_GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    return "generate_answer" if result.binary_score == "yes" else "rewrite_question"


REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying intent.\n"
    "Original question:\n-------\n{question}\n-------\n"
    "Produce an improved version of the question:"
)


def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    resp = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": resp.content}]}


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the retrieved context to answer the question concisely. "
    "If you don't know, say you don't know.\n\n"
    "Question: {question}\nContext: {context}"
)


def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    resp = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [resp]}


# ---------------------------------------------------------------------
# 5. Assemble the LangGraph workflow
# ---------------------------------------------------------------------
workflow = StateGraph(MessagesState)

# nodes
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# edges
workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END},
)

workflow.add_conditional_edges("retrieve", grade_documents)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# compile graph
graph = workflow.compile()

# ---------------------------------------------------------------------
# 6. Demo run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    user_question = "What does Lilian Weng say about types of reward hacking?"
    for chunk in graph.stream({"messages": [{"role": "user", "content": user_question}]}):
        for node, update in chunk.items():
            print(f"\n--- Update from node: {node} ---")
            print(update["messages"][-1].content)

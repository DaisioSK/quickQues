"""Layer 1: ingestion pipeline orchestration.

pipeline.py (integrator) wires PDFParser → Chunker → Embedder → VectorStore
and KeywordIndex into a single ``ingest(pdf_path)`` entry point.
"""

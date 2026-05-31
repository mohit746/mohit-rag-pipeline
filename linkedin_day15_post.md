I set out to build a production-grade RAG pipeline from scratch. 10 days in — here's what that actually looks like.

Not a tutorial. Not a wrapper around someone else's template.
A real system, designed decision by decision.

Here's what I architected:

→ Chunking strategies compared (fixed, recursive, paragraph) to understand how document structure affects retrieval quality
→ Embedding pipeline with Gemini + LangChain, moving from ChromaDB (local) to Pinecone (cloud-native, zero downtime)
→ pgvector on Neon as a PostgreSQL-native vector store — because sometimes your data already lives in a relational DB
→ Semantic Cache with Redis (Upstash) — one layer that slashes redundant LLM calls and cuts latency significantly
→ RAGAS evaluation framework to measure what actually matters: Faithfulness and Answer Relevancy

That last one changed how I think about RAG.

A pipeline that "works" and a pipeline that's faithful to its sources are two different things.
My Answer Relevancy hit 0.84. My Faithfulness is at 0.49 — and that number tells me exactly where to improve next.

That's the difference between shipping a demo and building a system.

Up next: LangChain agents and orchestration.

If you're building in the AI/ML space or who think in systems — let's connect.

#RAG #GenerativeAI #LLM #VectorSearch #Python #BuildInPublic #MLEngineering #AIArchitecture

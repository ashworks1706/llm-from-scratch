embeddings covers models that turn text into vectors for search and retrieval, and rerankers that score how relevant a document is to a query.

an embedding model is trained with a contrastive loss so matching pairs are close and others are far apart. a decoder llm can be turned into an embedder, matryoshka training lets you cut vectors to smaller sizes, and a reranker reads the query and document together.

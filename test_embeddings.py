
from modules.source_retriever import (
    generate_embeddings,
    create_faiss_index,
    retrieve_sources
)


# Sample research chunks
chunks = [

    {
        "filename": "AI_Research.pdf",
        "page_number": 1,
        "text": "Generative AI helps developers generate code automatically."
    },

    {
        "filename": "AI_Research.pdf",
        "page_number": 2,
        "text": "Artificial intelligence can improve software development productivity."
    },

    {
        "filename": "Education.pdf",
        "page_number": 3,
        "text": "Students use artificial intelligence to learn mathematics."
    }
]


# Step 1: Generate embeddings
embeddings = generate_embeddings(chunks)

print("Embedding shape:", embeddings.shape)


# Step 2: Create FAISS index
index = create_faiss_index(embeddings)

print("FAISS index created successfully!")

print("Total vectors stored:", index.ntotal)


# Step 3: Search relevant passages
query = "How does AI help software developers?"

results = retrieve_sources(
    query,
    chunks,
    index,
    top_k=2
)


# Step 4: Display results
print("\nRetrieved Sources:")

for result in results:

    print("\nFilename:", result["filename"])

    print("Page:", result["page_number"])

    print("Text:", result["text"])

    print(
        "Similarity:",
        round(result["similarity_score"], 4)
    )

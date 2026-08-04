import json
import numpy as np
from pathlib import Path
from embeddings import get_embedding
import matplotlib.pyplot as plt

cache_dir = Path()
cache_file = cache_dir / "embeddings_cache.json"

def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def build_embedding_matrix(documents):

    # Check if a cache already exists
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

        # Use the cache if it matches the current corpus
        
        if len(cache) == len(documents):
            cached_matrix = np.array(cache)

            if cached_matrix.shape[1] == len(get_embedding(documents[0]["text"], input_type="passage")):
                print("Loaded embeddings from cache.")
                return cached_matrix

        print("Corpus changed. Rebuilding embedding cache...")

    # Compute embeddings
    embeddings = []

    for doc in documents:
        embedding = get_embedding(doc["text"], input_type="passage")
        embeddings.append(embedding)

    # Save cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

    print("Saved embeddings to cache.")

    return np.array(embeddings)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, embedding_matrix, documents, top_k = 3):
    
    # Embed the query using query mode
    query_embedding = get_embedding(query, input_type="query")

    # Calculate similarity scores with every document
    scores = []

    for doc_embedding in embedding_matrix:
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append(score)

    # Get indices of top_k highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Return documents with their scores
    results = []

    for idx in top_indices:
        results.append((documents[idx], scores[idx]))

    return results

"""
Sanity Check 1: Output

[0.298] (mathematics) Probability measures the likelihood that an event will occur.
[0.283] (mathematics) Prime numbers have exactly two positive divisors: one and themselves.
[0.283] (astrology) Mercury retrograde is popularly believed to influence communication and travel.


Sanity Check 2: Output

[0.354] (mathematics) The Pythagorean theorem relates the sides of a right triangle.
[0.327] (sports) Basketball players score points by shooting the ball through the opponent's hoop.
[0.258] (mathematics) Prime numbers have exactly two positive divisors: one and themselves.

This is not working as expected, we would like to see cooking show up for check 1 and sports for check 2
"""

"""
Switching to API from offline - 

[0.422] (cooking) Baking bread requires yeast to ferment the dough and create air pockets.
[0.286] (cooking) Simmering soup over low heat allows flavors to blend gradually.
[0.265] (cooking) Boiling pasta in well-salted water improves its flavor and texture.
 --------------------------- 
[0.440] (sports) Football is played between two teams that compete to score goals.
[0.427] (sports) Basketball players score points by shooting the ball through the opponent's hoop.
[0.324] (sports) Tennis matches are won by earning enough sets according to the tournament rules.


"""

def pca_via_svd(data, n_components):
    # Center the data
    centered = data - data.mean(axis=0)

    # Apply SVD
    U, S, Vt = np.linalg.svd(centered)

    # Select top principal components
    components = Vt[:n_components]

    # Project data onto principal components
    projection = centered @ components.T

    return centered, projection, components

def visualize_pca(embedding_matrix, documents):
    # Apply PCA to embedding matrix
    centered, projection, components = pca_via_svd(
        embedding_matrix,
        n_components=2
    )

    # Extract topics for coloring
    topics = [doc["topic"] for doc in documents]

    # Create unique colors/labels
    unique_topics = list(set(topics))

    plt.figure(figsize=(8, 6))

    for topic in unique_topics:
        indices = [
            i for i, t in enumerate(topics)
            if t == topic
        ]

        plt.scatter(
            projection[indices, 0],
            projection[indices, 1],
            label=topic
        )


    plt.title("PCA Visualization of Document Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":

    documents = load_documents("documents.json")

    embedding_matrix = build_embedding_matrix(documents)

    results1 = search(
        "How does one bake bread?",
        embedding_matrix,
        documents,
        top_k=3
    )

    for doc, score in results1:
        print(f"[{score:.3f}] ({doc['topic']}) {doc['text']}")

    print (" --------------------------- ")

    results2 = search(
    "how do you score a goal?",
    embedding_matrix,
    documents,
    top_k=3
)

    for doc, score in results2:
        print(f"[{score:.3f}] ({doc['topic']}) {doc['text']}")

    visualize_pca(embedding_matrix, documents)
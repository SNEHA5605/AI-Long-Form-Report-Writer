
import gc
import re
import numpy as np
import faiss
import torch

from functools import lru_cache
from collections import Counter
from sentence_transformers import SentenceTransformer


# --------------------------------
# MODEL CONFIGURATION
# --------------------------------

torch.set_num_threads(1)


@lru_cache(maxsize=1)
def get_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )


# --------------------------------
# TEXT TOKENIZATION
# --------------------------------

STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in",
    "on", "and", "or", "to", "for", "this",
    "that", "are", "was", "were", "does",
    "do", "did", "how", "why", "when", "where",
    "which", "who", "can", "could", "would"
}


def tokenize(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return [
        word for word in words
        if word not in STOP_WORDS
    ]


# --------------------------------
# QUERY EXPANSION
# --------------------------------

def expand_query(query):

    terms = tokenize(query)

    expanded_terms = list(terms)

    objective_keywords = {
        "objective", "objectives",
        "aim", "aims",
        "purpose", "purposes",
        "goal", "goals"
    }

    if any(
        term in objective_keywords
        for term in terms
    ):

        expanded_terms.extend([
            "examine",
            "examines",
            "examined",
            "explore",
            "explores",
            "investigate",
            "investigates",
            "focus",
            "focuses"
        ])

    # Remove duplicate terms while preserving order
    return list(dict.fromkeys(expanded_terms))


# --------------------------------
# GENERATE EMBEDDINGS
# --------------------------------

def generate_embeddings(chunks):

    if not chunks:
        raise ValueError(
            "No text chunks available for embedding."
        )

    model = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    gc.collect()

    return embeddings


# --------------------------------
# CREATE FAISS INDEX
# --------------------------------

def create_faiss_index(embeddings):

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            "Invalid or empty embeddings."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# --------------------------------
# BM25 KEYWORD SEARCH
# --------------------------------

def calculate_bm25(
    query,
    chunks,
    k1=1.5,
    b=0.75
):

    documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    total_documents = len(documents)

    document_lengths = [
        len(document)
        for document in documents
    ]

    average_length = (
        sum(document_lengths) / total_documents
        if total_documents else 0
    )

    document_frequencies = Counter()

    for document in documents:

        document_frequencies.update(
            set(document)
        )

    query_terms = expand_query(query)

    scores = []

    for document, document_length in zip(
        documents,
        document_lengths
    ):

        frequencies = Counter(document)

        score = 0.0

        for term in query_terms:

            frequency = frequencies.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequencies[term]

            idf = np.log(
                1 + (
                    total_documents - df + 0.5
                ) / (
                    df + 0.5
                )
            )

            denominator = frequency + k1 * (
                1 - b + b * document_length / (
                    average_length or 1
                )
            )

            score += idf * (
                frequency * (k1 + 1)
            ) / denominator

        scores.append(score)

    return np.asarray(
        scores,
        dtype="float32"
    )


# --------------------------------
# NORMALIZE SCORES
# --------------------------------

def normalize_scores(scores):

    minimum = float(np.min(scores))

    maximum = float(np.max(scores))

    if maximum == minimum:

        return np.zeros_like(scores)

    return (
        (scores - minimum) /
        (maximum - minimum)
    )


# --------------------------------
# SECTION RELEVANCE
# --------------------------------

def calculate_section_bonus(
    chunk,
    query
):

    section = chunk.get(
        "section",
        "General Content"
    ).upper()

    query_terms = set(tokenize(query))

    bonus = 0.0

    # Research objective questions
    objective_terms = {
        "objective", "objectives",
        "aim", "aims",
        "purpose", "purposes",
        "goal", "goals"
    }

    if query_terms.intersection(objective_terms):

        if section == "ABSTRACT":
            bonus += 0.30

        elif section == "INTRODUCTION":
            bonus += 0.10

        elif section == "CONCLUSION":
            bonus += 0.08

    # Research methodology questions
    if query_terms.intersection({
        "method", "methods",
        "methodology", "approach"
    }):

        if "METHODOLOGY" in section:
            bonus += 0.20

    # Research finding questions
    if query_terms.intersection({
        "result", "results",
        "finding", "findings"
    }):

        if section in {
            "RESULTS",
            "DISCUSSION",
            "CONCLUSION"
        }:
            bonus += 0.20

    # Penalize reference lists
    if section == "REFERENCES":
        bonus -= 0.40

    # Penalize unclassified content containing
    # journal publication information
    text = chunk["text"].lower()

    if section == "GENERAL CONTENT":

        metadata_terms = [
            "issn",
            "volume",
            "journal",
            "www.",
            "copyright"
        ]

        if any(
            term in text
            for term in metadata_terms
        ):

            bonus -= 0.20

    return bonus


# --------------------------------
# HYBRID SOURCE RETRIEVAL
# --------------------------------

def retrieve_sources(
    query,
    chunks,
    index,
    top_k=5
):

    if not query.strip():

        raise ValueError(
            "Search query cannot be empty."
        )

    if index.ntotal == 0:
        return []

    if len(chunks) != index.ntotal:

        raise ValueError(
            "Chunks and FAISS index do not match. "
            "Please process the documents again."
        )

    model = get_embedding_model()

    # Step 1: Semantic search

    query_embedding = model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Retrieve scores for every chunk
    semantic_scores, indices = index.search(
        query_embedding,
        index.ntotal
    )

    # Step 2: BM25 keyword search

    keyword_scores = calculate_bm25(
        query,
        chunks
    )

    # Reorder semantic scores according to
    # the original chunk positions

    semantic_by_index = np.zeros(
        len(chunks),
        dtype="float32"
    )

    for score, idx in zip(
        semantic_scores[0],
        indices[0]
    ):

        if idx != -1:

            semantic_by_index[int(idx)] = score

    # Step 3: Normalize both scores

    normalized_semantic = normalize_scores(
        semantic_by_index
    )

    normalized_keyword = normalize_scores(
        keyword_scores
    )

    # Step 4: Combine semantic, keyword,
    # and section relevance scores

    final_scores = []

    for i, chunk in enumerate(chunks):

        section_bonus = calculate_section_bonus(
            chunk,
            query
        )

        final_score = (
            0.40 * normalized_semantic[i]
            + 0.45 * normalized_keyword[i]
            + section_bonus
        )

        final_scores.append(
            max(0.0, min(1.0, final_score))
        )

    # Step 5: Rank chunks by hybrid relevance

    ranked_indices = np.argsort(
        final_scores
    )[::-1]

    results = []

    for idx in ranked_indices[:top_k]:

        result = chunks[int(idx)].copy()

        result["similarity_score"] = float(
            semantic_by_index[idx]
        )

        result["keyword_score"] = float(
            keyword_scores[idx]
        )

        result["relevance_score"] = float(
            final_scores[idx]
        )

        results.append(result)

    return results

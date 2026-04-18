"""
embedding_utils.py
------------------
Utility functions for exploring static word embeddings (Word2Vec / GloVe).

Covers:
  - Training a tiny Word2Vec model from scratch with gensim
  - Analogy arithmetic: King - Man + Woman = ?
  - Nearest-neighbor search in vector space
  - 2D PCA projection for visualization

Usage:
    from src.embedding_utils import train_word2vec, analogy, plot_embeddings
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Optional


# -----------------------------------------------------------------------
# Word2Vec training (via gensim)
# -----------------------------------------------------------------------

def train_word2vec(
    sentences: List[List[str]],
    vector_size: int = 50,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 100,
    seed: int = 42,
):
    """
    Train a Word2Vec model on a list of tokenized sentences.

    Parameters
    ----------
    sentences   : list of tokenized sentences
    vector_size : dimensionality of the word vectors
    window      : context window size
    min_count   : ignore words that appear fewer than this many times
    epochs      : number of training passes
    seed        : random seed for reproducibility

    Returns
    -------
    gensim.models.Word2Vec model
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("Please install gensim: pip install gensim")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        seed=seed,
        epochs=epochs,
    )
    print(f"Word2Vec trained. Vocabulary size: {len(model.wv.key_to_index)} words")
    return model


# -----------------------------------------------------------------------
# Analogy arithmetic
# -----------------------------------------------------------------------

def analogy(
    model,
    positive: List[str],
    negative: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Solve word analogies using vector arithmetic.

    Example:
        analogy(model, positive=["king", "woman"], negative=["man"])
        → should return "queen" near the top

    Parameters
    ----------
    model    : a fitted gensim Word2Vec model (or KeyedVectors)
    positive : words to ADD in vector space
    negative : words to SUBTRACT from vector space
    top_k    : number of results to return

    Returns
    -------
    list of (word, cosine_similarity) tuples
    """
    wv = model.wv if hasattr(model, "wv") else model
    results = wv.most_similar(positive=positive, negative=negative, topn=top_k)
    return results


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / norm) if norm > 0 else 0.0


# -----------------------------------------------------------------------
# Nearest neighbour search
# -----------------------------------------------------------------------

def nearest_neighbors(model, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Return the top_k most similar words to `word`.

    Parameters
    ----------
    model : gensim Word2Vec model
    word  : the query word
    top_k : number of neighbours to return
    """
    wv = model.wv if hasattr(model, "wv") else model
    if word not in wv.key_to_index:
        print(f"'{word}' not in vocabulary.")
        return []
    return wv.most_similar(word, topn=top_k)


# -----------------------------------------------------------------------
# PCA visualization
# -----------------------------------------------------------------------

def plot_embeddings(
    model,
    words: Optional[List[str]] = None,
    title: str = "Word2Vec Embeddings — PCA Projection",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
):
    """
    Project word vectors to 2D using PCA and plot them.

    Parameters
    ----------
    model     : fitted gensim Word2Vec model
    words     : list of words to plot. If None, plots all words in vocabulary.
    title     : plot title
    save_path : if provided, saves the figure to this path
    figsize   : figure size in inches
    """
    from sklearn.decomposition import PCA

    wv = model.wv if hasattr(model, "wv") else model

    if words is None:
        words = list(wv.key_to_index.keys())

    # Filter to words actually in vocabulary
    words = [w for w in words if w in wv.key_to_index]
    if not words:
        print("No valid words found in vocabulary.")
        return

    vectors = np.array([wv[w] for w in words])

    # Reduce to 2D
    n_components = min(2, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(vectors)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=60, color="#185FA5")

    for i, word in enumerate(words):
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="#2C2C2A",
        )

    ax.set_title(title, fontsize=13, fontweight="medium", pad=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig, ax


# -----------------------------------------------------------------------
# GloVe loader (from raw .txt files)
# -----------------------------------------------------------------------

def load_glove_vectors(glove_path: str) -> dict:
    """
    Load pre-trained GloVe vectors from the standard .txt format.

    Download from: https://nlp.stanford.edu/projects/glove/
    Recommended file: glove.6B.50d.txt

    Parameters
    ----------
    glove_path : path to the .txt file

    Returns
    -------
    dict mapping word -> numpy vector
    """
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} GloVe vectors.")
    return embeddings


def glove_nearest_neighbors(
    embeddings: dict, word: str, top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find nearest neighbours in a GloVe embedding dict.

    Parameters
    ----------
    embeddings : dict from load_glove_vectors()
    word       : query word
    top_k      : number of results

    Returns
    -------
    list of (word, cosine_similarity) sorted descending
    """
    if word not in embeddings:
        print(f"'{word}' not in GloVe vocabulary.")
        return []

    query_vec = embeddings[word]
    sims = []
    for w, vec in embeddings.items():
        if w == word:
            continue
        sim = cosine_similarity(query_vec, vec)
        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

"""
bert_utils.py
-------------
Helper functions for extracting contextual embeddings from BERT and
running zero-shot / fine-tuned classification.

Key demos:
  - Show that "bank" gets a DIFFERENT vector in "river bank" vs "financial bank"
  - Compute cosine similarity between contextual embeddings
  - Simple sentiment classification pipeline

Usage:
    from src.bert_utils import get_word_embedding, compare_word_in_contexts
"""

import numpy as np
from typing import List, Tuple, Optional


# -----------------------------------------------------------------------
# Tokenization helpers
# -----------------------------------------------------------------------

def load_bert(model_name: str = "bert-base-uncased"):
    """
    Load a BERT tokenizer and model. Downloads on first call (~440 MB).

    Parameters
    ----------
    model_name : HuggingFace model identifier

    Returns
    -------
    (tokenizer, model) tuple
    """
    try:
        from transformers import BertTokenizer, BertModel
        import torch
    except ImportError:
        raise ImportError("Install transformers and torch: pip install transformers torch")

    print(f"Loading {model_name}... (first run downloads the model weights)")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model


# -----------------------------------------------------------------------
# Contextual embedding extraction
# -----------------------------------------------------------------------

def get_word_embedding(
    sentence: str,
    target_word: str,
    tokenizer,
    model,
    layer: int = -2,
) -> Optional[np.ndarray]:
    """
    Extract the contextual embedding for a specific word in a sentence.

    BERT tokenizes using WordPiece — a word may split into multiple subword
    tokens. This function averages the subword token embeddings.

    Parameters
    ----------
    sentence     : the full input sentence
    target_word  : the word whose embedding we want
    tokenizer    : BertTokenizer
    model        : BertModel
    layer        : which hidden layer to use. -1 = last layer, -2 = second-to-last.
                   Paper recommendation: layers -1 to -4 work well.

    Returns
    -------
    numpy array of shape (hidden_size,), or None if word not found.
    """
    import torch

    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states is a tuple: (embedding_layer, layer_1, ..., layer_12)
    hidden_states = outputs.hidden_states
    chosen_layer = hidden_states[layer][0]  # shape: (seq_len, hidden_size)

    # Find which token positions correspond to our target word
    target_lower = target_word.lower()
    matched_indices = []
    for i, token in enumerate(tokens):
        # WordPiece tokens that are part of target_word
        clean_token = token.replace("##", "")
        if clean_token in target_lower or target_lower.startswith(clean_token):
            matched_indices.append(i)

    if not matched_indices:
        # Fallback: fuzzy match
        for i, token in enumerate(tokens):
            if token.replace("##", "") in target_lower:
                matched_indices.append(i)

    if not matched_indices:
        print(f"Warning: '{target_word}' not found in tokens: {tokens}")
        return None

    # Average the subword token embeddings
    token_embeddings = chosen_layer[matched_indices]
    word_embedding = token_embeddings.mean(dim=0).numpy()
    return word_embedding


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(dot / norm) if norm > 0 else 0.0


# -----------------------------------------------------------------------
# The core demo: same word, different vectors
# -----------------------------------------------------------------------

def compare_word_in_contexts(
    word: str,
    sentences: List[str],
    tokenizer,
    model,
    layer: int = -2,
    verbose: bool = True,
) -> np.ndarray:
    """
    The main BERT demo: extract the embedding for `word` in each sentence
    and show that the vectors differ across contexts.

    Parameters
    ----------
    word      : the ambiguous word (e.g. "bank")
    sentences : list of sentences containing the word
    tokenizer : BertTokenizer
    model     : BertModel
    layer     : which hidden layer to extract from
    verbose   : if True, prints similarity matrix

    Returns
    -------
    numpy array of shape (len(sentences), hidden_size)
    """
    embeddings = []
    for sent in sentences:
        emb = get_word_embedding(sent, word, tokenizer, model, layer=layer)
        if emb is not None:
            embeddings.append(emb)
        else:
            embeddings.append(np.zeros(768))  # fallback

    embeddings = np.array(embeddings)

    if verbose:
        print(f"\nContextual embeddings for '{word}'")
        print("=" * 60)
        for i, sent in enumerate(sentences):
            print(f"  [{i+1}] {sent}")
        print()

        print("Pairwise cosine similarities:")
        print("(1.0 = identical meaning, 0.0 = completely different)")
        print("-" * 60)
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                print(f"  Sentence {i+1} vs Sentence {j+1}: {sim:.4f}")

        print()
        print("KEY INSIGHT: If BERT truly captures context, sentences where")
        print(f"'{word}' has different meanings should have LOWER similarity.")

    return embeddings


# -----------------------------------------------------------------------
# Sentiment classification pipeline
# -----------------------------------------------------------------------

def bert_sentiment(sentences: List[str]) -> List[dict]:
    """
    Run zero-shot sentiment classification using a fine-tuned BERT pipeline.

    Uses 'distilbert-base-uncased-finetuned-sst-2-english' — a smaller,
    faster model suitable for demonstration purposes.

    Parameters
    ----------
    sentences : list of strings

    Returns
    -------
    list of dicts: [{"label": "POSITIVE", "score": 0.99}, ...]
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    print("Loading sentiment pipeline (DistilBERT fine-tuned on SST-2)...")
    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
    )

    results = clf(sentences)
    for sent, res in zip(sentences, results):
        emoji = "✅" if res["label"] == "POSITIVE" else "❌"
        print(f"  {emoji}  [{res['label']:8s} {res['score']:.2f}]  {sent}")

    return results


# -----------------------------------------------------------------------
# Attention visualization helper
# -----------------------------------------------------------------------

def get_attention_weights(sentence: str, tokenizer, model) -> Tuple[List[str], np.ndarray]:
    """
    Extract raw attention weights from all heads in the last BERT layer.

    Useful for visualizing what the model is "looking at" for each token.

    Parameters
    ----------
    sentence  : input sentence
    tokenizer : BertTokenizer
    model     : BertModel (must be loaded with output_attentions=True)

    Returns
    -------
    (tokens, attention_matrix)
    tokens           : list of string tokens
    attention_matrix : numpy array, shape (num_heads, seq_len, seq_len)
                       averaged across heads → (seq_len, seq_len)
    """
    import torch
    from transformers import BertModel, BertTokenizer

    # Reload model with attentions enabled if needed
    if not hasattr(model.config, "output_attentions") or not model.config.output_attentions:
        model = BertModel.from_pretrained(
            model.config.name_or_path, output_attentions=True
        )
        model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (batch, heads, seq, seq) for each layer
    # Use the last layer
    last_layer_attn = outputs.attentions[-1][0]  # (heads, seq, seq)
    avg_attn = last_layer_attn.mean(dim=0).numpy()  # (seq, seq)

    return tokens, avg_attn

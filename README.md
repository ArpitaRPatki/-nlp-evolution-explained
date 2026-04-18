#  NLP Evolution: From N-grams to BERT

> **Companion code for the Medium article:**  
> *"From Counting Words to Understanding Meaning: A Brief History of NLP"*

This repository walks through every concept in the article with **runnable Python code** — from a hand-built N-gram model all the way to loading BERT for sentence classification. Each notebook is self-contained and heavily commented so you can follow along even if you're just starting out.

---

##  Project Structure

```
nlp-evolution-explained/
│
├── notebooks/
│   ├── 01_ngrams_and_smoothing.ipynb       ← N-grams, sparsity, Laplace smoothing
│   ├── 02_word2vec_and_glove.ipynb         ← Static embeddings, the King-Queen analogy
│   └── 03_bert_contextual_embeddings.ipynb ← BERT, attention, contextual vectors
│
├── src/
│   ├── ngram_model.py                      ← Clean N-gram class you can import
│   ├── embedding_utils.py                  ← Helpers for Word2Vec exploration
│   └── bert_utils.py                       ← BERT tokenization + inference helpers
│
├── data/
│   └── sample_corpus.txt                   ← Small toy corpus used across notebooks
│
├── outputs/
│   └── (generated plots saved here)
│
├── requirements.txt
└── README.md
```

---

##  Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nlp-evolution-explained.git
cd nlp-evolution-explained
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook notebooks/
```

Open them **in order** — each one builds on the previous.

---

##  What's Inside Each Notebook

### `01_ngrams_and_smoothing.ipynb`
- Build a unigram, bigram, and trigram model from scratch
- Visualize the sparsity problem with a real probability table
- Implement Laplace smoothing and see why it's a patch, not a solution
- Generate text from an N-gram model

### `02_word2vec_and_glove.ipynb`
- Train a tiny Word2Vec model using `gensim`
- Reproduce the `King − Man + Woman ≈ Queen` arithmetic
- Load pre-trained GloVe vectors and compare nearest neighbors
- Plot a 2D PCA projection of word vectors to *see* semantic clusters

### `03_bert_contextual_embeddings.ipynb`
- Tokenize sentences using `BertTokenizer`
- Extract contextual embeddings for the same word in different sentences
- Show that "bank" gets a **different vector** in "river bank" vs "financial bank"
- Fine-tune BERT for simple sentiment classification on a small dataset

---

##  Key Concept at a Glance

| Era | Model | Representation | Context-aware? |
|-----|-------|----------------|----------------|
| Statistical | N-gram | Count tables |  (short window only) |
| Vector Space | Word2Vec / GloVe | Dense static vectors |  (one vector per word) |
| Contextual | ELMo / BERT | Dynamic vectors |  (full sentence) |

---

## 🛠 Requirements

See `requirements.txt`. Main dependencies:
- `numpy`, `matplotlib`, `pandas`
- `nltk` — tokenization and N-gram utilities
- `gensim` — Word2Vec training
- `transformers` — Hugging Face BERT
- `torch` — PyTorch backend
- `scikit-learn` — PCA for visualization
- `jupyter`

---

##  Read the Full Article

 [Medium Article Link — add yours here]

---

##  Contributing

Found a bug or want to add ELMo comparisons? PRs are welcome. Open an issue first if it's a big change.

---

##  License

MIT — use freely, attribution appreciated.

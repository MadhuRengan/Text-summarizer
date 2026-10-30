# Document Search and Summarization Using LLMs

A system that searches a news corpus (CNN/DailyMail) and summarizes the results with a Large Language Model. It combines traditional keyword search (BM25) with dense embedding search (FAISS), merges the two with Reciprocal Rank Fusion, and summarizes the top results at a length chosen by the user. It includes an evaluation pipeline (Recall@k, MRR, ROUGE, human scoring) and a Streamlit interface.

## Features

- **Data preparation:** cleaning, deduplication and length filtering of 20,000 CNN/DailyMail articles
- **Hybrid search:** BM25 (keywords) + `BAAI/bge-small-en-v1.5` embeddings in FAISS, fused with Reciprocal Rank Fusion. Search method can be switched between `bm25`, `dense` and `hybrid`
- **Summarization:** open-source `Qwen/Qwen2.5-3B-Instruct`, run locally, with a user-specified summary length (short, medium, long, or any number of words)
- **Evaluation:** Recall@1, Recall@5 and MRR for search; ROUGE (against a Lead-3 baseline) and human scoring for summaries
- **Interface (bonus):** Streamlit app with query suggestions, pagination and an adjustable summary length

## Project files

```
.
├── doc_search_summarization.ipynb   # data prep, indexing, search, summarizer, evaluation (rename to match your notebook)
├── app.py                           # Streamlit interface
├── requirements.txt                 # Python dependencies
├── README.md
└── (generated when you run the notebook)
    ├── docs.pkl                     # cleaned corpus + held-out test ids
    ├── cnn_dm.faiss                 # dense (FAISS) index
    ├── bm25.pkl                     # BM25 index
    ├── eval_queries.json            # LLM-generated test queries
    ├── retrieval_results.csv        # Recall@k / MRR table
    ├── rouge_results.csv            # ROUGE table
    └── human_eval_scored.csv        # human scores for 10 summaries
```

## Requirements

- Python 3.10 or newer
- A GPU is strongly recommended for embedding and for the LLM (a free Colab **T4** is enough). The code also runs on CPU, but indexing and summarization are much slower
- About 6 GB of disk space and GPU memory for the Qwen2.5-3B-Instruct model (first run downloads it). If memory is short, use `Qwen/Qwen2.5-1.5B-Instruct` instead
- No API key is needed. The default setup uses only open-source models

## Setup

### Option A: Google Colab (easiest)

1. Open the notebook in Colab and choose **Runtime > Change runtime type > T4 GPU**.
2. Run the install cell, then **restart the session**:
   ```
   !pip install -U datasets sentence-transformers faiss-cpu rank_bm25 rouge-score transformers accelerate streamlit tqdm numpy pandas
   ```
3. Run the notebook cells from top to bottom (see "Run order" below).

### Option B: Local machine

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook                  # then open the notebook
```

For a GPU, install the CUDA build of PyTorch first (see https://pytorch.org), and use `faiss-gpu` instead of `faiss-cpu` only if you need it. `faiss-cpu` is fine for 20,000 documents.

## Run order (notebook)

1. **Imports and config:** libraries and the random seed (42).
2. **Stage 1, data preparation:** loads CNN/DailyMail, samples 25,000 articles, cleans them, removes short, long and duplicate articles, keeps 20,000, holds out 200 test documents, and saves `docs.pkl`.
3. **Stage 2, indexing:** embeds all articles with `bge-small-en-v1.5`, builds and saves the FAISS index (`cnn_dm.faiss`) and the BM25 index (`bm25.pkl`).
4. **Stage 3, search:** `dense_search`, `bm25_search`, `rrf_fuse` and `search(query, n, mode)`.
5. **Stage 4, summarizer:** loads the local LLM and defines `summarize(results, length)` and `search_and_summarize(query, n, length, mode)`.
6. **Stage 5, evaluation:** generates test queries, computes Recall@k and MRR for the three search modes, ROUGE against the Lead-3 baseline, and exports summaries for human scoring.

Quick usage example after Stage 4:

```python
results, summary, search_secs, llm_secs = search_and_summarize(
    "senate rejects balanced budget amendment", n=3, length="medium", mode="hybrid"
)
print(summary)
```

`length` can be `"short"` (about 50 words), `"medium"` (about 120), `"long"` (about 250) or an integer number of words.

## Running the Streamlit interface

`app.py` loads `docs.pkl`, `cnn_dm.faiss` and `bm25.pkl`, so run Stages 1 and 2 first (or copy those files next to `app.py`).

```bash
streamlit run app.py
```

Then open the local URL that Streamlit prints (usually http://localhost:8501). The first load takes one to three minutes while the models load.

In the app you can:

- type a query (matching topics are suggested as you type; press Enter to update them)
- choose the search method: hybrid, dense or BM25
- set the summary length (30 to 250 words) and how many top results to summarize
- page through the top 30 results, 5 per page, and expand any article

### Running the app from Colab

Colab has no public URL, so use a tunnel:

```
!pip install -q streamlit
!nohup streamlit run app.py --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false > /content/streamlit.log 2>&1 &
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8501 --no-autoupdate
```

Open the `https://....trycloudflare.com` link that is printed. Keep the cell running while you use the app. The link is temporary.

## Results summary

| Search method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 | 0.98 | 1.00 | 0.987 |
| Dense | 0.94 | 1.00 | 0.970 |
| Hybrid (RRF) | 0.96 | 1.00 | 0.977 |

| Summary system | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-3 baseline | 0.387 | 0.151 | 0.241 |
| Qwen2.5-3B-Instruct | 0.363 | 0.118 | 0.237 |

Human evaluation (10 summaries, 1 to 5 scale): relevance 3.8, coherence 3.8, faithfulness 3.7.

See the report for the full methodology, interpretation and limitations. The retrieval test uses 50 queries, ROUGE uses 30 documents and the human evaluation has one rater, so treat these numbers as indicative.

## Configuration notes

- **Random seed:** 42 is used for sampling, shuffling and the test set, so results are reproducible.
- **LLM choice:** the model name is set in one place (`LLM_NAME` in `app.py`, and `LOCAL_MODEL` in the notebook). Change it to try another instruction-tuned model.
- **Using a hosted LLM instead (optional):** the summarizer only needs an object that works in a LangChain chain (`prompt | llm | StrOutputParser()`). You can replace the local model with `ChatGoogleGenerativeAI` (Gemini) or `ChatOpenAI`. Set the key through an environment variable or Colab Secrets. **Never hard-code an API key in the notebook or commit it to the repository.** Free API tiers have strict daily limits, which is why the default is a local model.
- **Corpus size:** change `N_DOCS` in Stage 1 to use fewer or more articles. Indexes must be rebuilt after changing it.

## Scalability notes

- Indexes are built once and saved to disk. Embedding is batched.
- For millions of documents, replace `IndexFlatIP` with an approximate FAISS index (HNSW or IVF) and move BM25 to an inverted-index engine such as Elasticsearch or OpenSearch.
- LLM generation is the slowest step. The app summarizes only the top N results, truncates each article to 400 words and caches results by query and length.

## Known limitations

- Test queries are LLM-generated and share many words with their source article, which favors BM25. Evaluation with paraphrased and human-written queries is future work.
- The local 3B model can add unsupported details to summaries (faithfulness is the weakest human-evaluation score).
- No fine-tuning and no cross-encoder reranker are included.

## Data and model credits

- Dataset: [CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail)
- Embeddings: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- LLM: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

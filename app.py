import re, time, pickle
import numpy as np
import faiss, torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="News Search & Summarizer", layout="wide")
PAGE_SIZE, TOP_K = 5, 30                                   # 5 results per page, 30 retrieved per query
LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ---------- Load everything once (cached across reruns) ----------
@st.cache_resource(show_spinner="Loading indexes and models (first run takes a minute)...")
def load_all():
    docs = pickle.load(open("docs.pkl", "rb"))["docs"]
    index = faiss.read_index("cnn_dm.faiss")
    bm25 = pickle.load(open("bm25.pkl", "rb"))
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    lm = AutoModelForCausalLM.from_pretrained(LLM_NAME, dtype=torch.float16, device_map="auto")
    suggestions = [d["summary"].split("\n")[0].strip() for d in docs]   # first highlight line of each article
    return docs, index, bm25, embedder, tok, lm, suggestions

docs, index, bm25, embedder, tok, lm, suggestions = load_all()
tokenize = lambda s: re.findall(r"\w+", s.lower())

# ---------- Search ----------
def dense_search(q, k):
    v = embedder.encode([q], normalize_embeddings=True).astype("float32")
    s, i = index.search(v, k)
    return [(int(a), float(b)) for a, b in zip(i[0], s[0])]

def bm25_search(q, k):
    sc = bm25.get_scores(tokenize(q))
    top = np.argsort(sc)[::-1][:k]
    return [(int(i), float(sc[i])) for i in top]

def rrf(lists, c=60):                                       # Reciprocal Rank Fusion
    f = {}
    for res in lists:
        for r, (d, _) in enumerate(res, 1):
            f[d] = f.get(d, 0) + 1 / (c + r)
    return sorted(f.items(), key=lambda x: -x[1])

@st.cache_data(show_spinner=False)
def run_search(q, mode):
    t0 = time.time()
    if mode == "Dense (embeddings)":
        r = dense_search(q, 50)
    elif mode == "BM25 (keywords)":
        r = bm25_search(q, 50)
    else:
        r = rrf([dense_search(q, 50), bm25_search(q, 50)])
    return r[:TOP_K], time.time() - t0

# ---------- Summarization (local LLM, cached per query/length/results) ----------
@st.cache_data(show_spinner=False)
def run_summary(q, words, ids):
    content = "\n\n".join(f"Article {n+1}: " + " ".join(docs[i]["text"].split()[:400]) for n, i in enumerate(ids))
    prompt = (f"You are a news summarizer. Summarize the article(s) below in about {words} words "
              f"(do not exceed {words} words). Use only information from the text, be factual and coherent, "
              f"and if there are several articles, combine them into one summary instead of summarizing each separately.\n\n"
              f"{content}\n\nSummary:")
    t0 = time.time()
    inputs = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(lm.device)
    with torch.no_grad():
        out = lm.generate(**inputs, max_new_tokens=int(words * 1.8) + 20, do_sample=False)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, time.time() - t0

# ---------- UI ----------
st.title("📰 News Search & Summarization")
st.session_state.setdefault("page", 0)

with st.sidebar:
    mode = st.radio("Search method", ["Hybrid (BM25 + embeddings)", "Dense (embeddings)", "BM25 (keywords)"])
    words = st.slider("Summary length (words)", 30, 250, 100, 10)
    n_sum = st.slider("Summarize top N results", 1, 5, 3)
    show_summary = st.toggle("Generate summary", value=True)

st.text_input("Search the news corpus", key="query", placeholder="e.g. senate vote on balanced budget amendment")
q = st.session_state.get("query", "").strip()

# query auto-suggestions (appear after you press Enter or click out of the box)
if len(q) >= 3:
    matches = [s for s in suggestions if q.lower() in s.lower()][:5]
    if matches and q not in matches:
        st.caption("Suggestions:")
        for n, s in enumerate(matches):
            st.button(s, key=f"sg{n}", on_click=lambda s=s: st.session_state.update(query=s))

def go(delta):
    st.session_state.page += delta

if q:
    if st.session_state.get("last") != (q, mode):            # new query or mode: back to page 1
        st.session_state.page, st.session_state.last = 0, (q, mode)
    results, secs = run_search(q, mode)
    st.write(f"**{len(results)} results** retrieved in {secs*1000:.0f} ms")

    if show_summary:
        with st.spinner("Summarizing..."):
            summ, lsecs = run_summary(q, words, tuple(d for d, _ in results[:n_sum]))
        st.subheader(f"Summary of top {n_sum} results")
        st.info(summ)
        st.caption(f"{len(summ.split())} words (target {words}) · generated in {lsecs:.1f} s")

    n_pages = (len(results) + PAGE_SIZE - 1) // PAGE_SIZE
    c1, c2, c3 = st.columns([1, 1, 6])
    c1.button("◀ Prev", on_click=go, args=(-1,), disabled=st.session_state.page <= 0)
    c2.button("Next ▶", on_click=go, args=(1,), disabled=st.session_state.page >= n_pages - 1)
    c3.write(f"Page {st.session_state.page + 1} of {n_pages}")

    start = st.session_state.page * PAGE_SIZE
    for rank, (d, score) in enumerate(results[start:start + PAGE_SIZE], start + 1):
        with st.container(border=True):
            st.markdown(f"**#{rank}** · doc {d} · score {score:.4f}")
            st.write(" ".join(docs[d]["text"].split()[:60]) + " ...")
            with st.expander("Full article"):
                st.write(docs[d]["text"])

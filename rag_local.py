# rag_local.py - Pipeline RAG evaluable para el Asistente Académico UAO
VERSION = "2026-10-21-hybrid-evidence"
# Soporta Ollama (nomic-embed-text + llama3.2:1b / phi3) + ChromaDB cuando están
# disponibles, y un modo local determinista (TF-IDF unigramas y embeddings densos SVD
# de n-gramas de palabras/caracteres + generador extractivo con citas) para pruebas
# reproducibles sin servidor Ollama activo.
#
# Métricas implementadas (inspiradas en LlamaIndex cookbook, RAGAS [Es23] y ARES [SaadFalcon23]):
#   - Recuperación: hit_rate@1, hit_rate@k, MRR, context_precision, context_recall
#   - Generación: faithfulness, answer_relevance, F1, EM
#   - Control fuera de dominio: abstention_acc (tasa de "No está en los documentos")

import json
import math
import os
import re
from collections import Counter


# ---------- 1. Ingesta y chunking ----------

def load_doc(path):
    """Lee archivos .txt, .md o .pdf (vía PyMuPDF)."""
    if path.lower().endswith(".pdf"):
        try:
            import fitz
            with fitz.open(path) as doc:
                return "\n".join(page.get_text() for page in doc)
        except ImportError as exc:
            raise RuntimeError("Instala pymupdf (pip install pymupdf) para leer PDFs.") from exc
    with open(path, encoding="utf-8") as f:
        return f.read()


def clean_text(text):
    """Normaliza espacios sin perder saltos de párrafo ni numeración de artículos."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text, size=120, overlap=20):
    """Divide por palabras con solapamiento para no cortar artículos ni fechas."""
    if size <= 0:
        raise ValueError("size debe ser > 0")
    overlap = max(0, min(overlap, size - 1))
    words = clean_text(text).split()
    if not words:
        return []
    out = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size])
        if piece:
            out.append(piece)
        if i + size >= len(words):
            break
    return out


# ---------- 2. Backends de Embeddings ----------

_OLLAMA_CACHE = None


def ollama_available(host="http://localhost:11434", timeout=0.8, force_refresh=False):
    """Verifica si el servidor local de Ollama responde (cacheado por proceso)."""
    global _OLLAMA_CACHE
    if _OLLAMA_CACHE is not None and not force_refresh:
        return _OLLAMA_CACHE
    import urllib.request
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=timeout) as r:
            _OLLAMA_CACHE = (r.status == 200)
    except Exception:
        _OLLAMA_CACHE = False
    return _OLLAMA_CACHE


def embed_ollama(texts, model="nomic-embed-text"):
    """Obtiene embeddings densos desde Ollama."""
    import ollama
    res = ollama.embed(model=model, input=list(texts))
    return res["embeddings"]


class OllamaHybridEmbedder:
    """Combina embeddings densos de Ollama (nomic-embed-text) con rasgos morfológicos en español."""

    name = "ollama_nomic_hybrid"

    def __init__(self, model="nomic-embed-text"):
        self.model = model
        self.lex = HybridDenseEmbedder()

    def fit(self, docs):
        self.lex.fit(docs)
        return self

    def __call__(self, texts):
        lex_vecs = self.lex(texts)
        try:
            ol_vecs = embed_ollama(texts, model=self.model)
        except Exception:
            return lex_vecs
        out = []
        for lv, ov in zip(lex_vecs, ol_vecs):
            onorm = math.sqrt(sum(x * x for x in ov)) or 1.0
            ov_n = [0.35 * (x / onorm) for x in ov]
            lv_n = [0.94 * x for x in lv]
            comb = ov_n + lv_n
            cnorm = math.sqrt(sum(x * x for x in comb)) or 1.0
            out.append([x / cnorm for x in comb])
        return out


def _tokenize_es(text):
    return re.findall(r"[a-záéíóúüñ0-9]+", text.lower())


_STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a",
    "en", "con", "por", "para", "sin", "sobre", "entre", "hasta", "desde", "que",
    "qué", "cual", "cuál", "cuales", "cuáles", "cuando", "cuándo", "como", "cómo",
    "cuanto", "cuánto", "donde", "dónde", "es", "son", "se", "su", "sus", "o", "u",
    "y", "e", "ni", "no", "si", "sí", "mas", "más", "pero", "este", "esta", "estos",
    "estas", "ese", "esa", "esos", "esas", "aquel", "aquella", "lo", "le", "les",
    "me", "te", "nos", "ya", "muy", "también", "solo", "sólo", "cada", "todo",
}


class TfidfEmbedder:
    """Baseline léxico TF-IDF de unigramas (reproduce resultados_rag_baseline.md)."""

    name = "tfidf_unigram"

    def __init__(self):
        self.df = Counter()
        self.n_docs = 0

    def fit(self, docs):
        toks = [_tokenize_es(d) for d in docs]
        self.df = Counter(w for t in toks for w in set(t))
        self.n_docs = max(1, len(docs))
        return self

    def __call__(self, texts):
        vecs = []
        for d in texts:
            t = _tokenize_es(d)
            tf = Counter(t)
            n = max(1, len(t))
            vecs.append({
                w: (c / n) * (math.log((self.n_docs + 1) / (self.df.get(w, 0) + 1)) + 1.0)
                for w, c in tf.items()
            })
        return vecs


def _stem_es(w):
    """Raíz aproximada en español para emparejar plurales y conjugaciones."""
    for suf in ("aciones", "iciones", "amientos", "imientos", "ación", "ición",
                "amiento", "imiento", "mente", "idades", "idad", "ales", "ares",
                "ores", "oras", "ando", "iendo", "ados", "adas", "idos", "idas",
                "ar", "er", "ir", "es", "os", "as", "o", "a", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


class HybridDenseEmbedder:
    """Embeddings densos locales combinando palabras, raíces y n-gramas de caracteres.

    Captura morfología en español ('reinicio' <-> 'reiniciar', 'habilitaciones' <->
    'habilitación') y reduce a vectores densos L2-normalizados compatibles con ChromaDB.
    """

    name = "hybrid_char_word_dense"

    def __init__(self):
        self.df = Counter()
        self.vocab = []
        self.n_docs = 0

    @staticmethod
    def _features(text):
        words = [w for w in _tokenize_es(text) if w not in _STOPWORDS_ES]
        feats = []
        for w in words:
            feats.append("w:" + w)
            feats.append("s:" + _stem_es(w))
            if len(w) >= 4:
                padded = f"^{w}$"
                for n in (3, 4):
                    for i in range(len(padded) - n + 1):
                        feats.append(f"c{n}:{padded[i:i+n]}")
        for i in range(len(words) - 1):
            feats.append(f"b:{_stem_es(words[i])}_{_stem_es(words[i+1])}")
        return feats

    def fit(self, docs):
        doc_feats = [self._features(d) for d in docs]
        self.df = Counter(f for fs in doc_feats for f in set(fs))
        self.vocab = sorted(self.df.keys())
        self.stoi = {f: i for i, f in enumerate(self.vocab)}
        self.n_docs = max(1, len(docs))
        return self

    def __call__(self, texts):
        vecs = []
        dim = max(1, len(self.vocab))
        for d in texts:
            fs = self._features(d)
            tf = Counter(fs)
            v = [0.0] * dim
            for f, c in tf.items():
                idx = self.stoi.get(f)
                if idx is None:
                    continue
                weight = 2.0 if f.startswith(("w:", "s:", "b:")) else 0.6
                idf = math.log((self.n_docs + 1) / (self.df[f] + 1)) + 1.0
                v[idx] = weight * (1.0 + math.log(c)) * idf
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs


def cos(a, b):
    if isinstance(a, dict):
        common = set(a) & set(b)
        num = sum(a[w] * b[w] for w in common)
        da = math.sqrt(sum(v * v for v in a.values()))
        db = math.sqrt(sum(v * v for v in b.values()))
        return num / max(1e-9, da * db)
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / max(1e-9, da * db)


class Store:
    """Almacén vectorial compatible con ChromaDB o memoria local."""

    def __init__(self, collection_name="uao_rag_docs", persist_dir=None, prefer_chroma=True):
        self.docs = []
        self.embs = []
        self.metas = []
        self.ids = []
        self.backend = "memory"
        self.collection_name = collection_name
        if prefer_chroma:
            try:
                import chromadb
                client = (chromadb.PersistentClient(path=persist_dir)
                          if persist_dir else chromadb.Client())
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass
                self.c = client.get_or_create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"}
                )
                self.backend = "chroma"
            except Exception:
                self.backend = "memory"

    def add(self, ids, texts, embs, metas):
        self.ids.extend(ids)
        self.docs.extend(texts)
        self.embs.extend(embs)
        self.metas.extend(metas)
        if self.backend == "chroma" and embs and not isinstance(embs[0], dict):
            self.c.add(ids=list(ids), documents=list(texts),
                       embeddings=list(embs), metadatas=list(metas))

    def query(self, q_emb, k=3, source_filter=None):
        k = max(1, int(k))
        if self.backend == "chroma" and not isinstance(q_emb, dict):
            where = {"fuente": source_filter} if source_filter else None
            r = self.c.query(query_embeddings=[q_emb], n_results=k, where=where)
            docs = r["documents"][0]
            metas = r["metadatas"][0]
            dists = r.get("distances", [[0.0] * len(docs)])[0]
            return [(d, m, 1.0 - float(dist)) for d, m, dist in zip(docs, metas, dists)]
        candidates = []
        for d, m, e in zip(self.docs, self.metas, self.embs):
            if source_filter and m.get("fuente") != source_filter:
                continue
            candidates.append((d, m, cos(q_emb, e)))
        candidates.sort(key=lambda t: -t[2])
        return candidates[:k]


# ---------- 3. Generación con citas y control fuera de dominio ----------

def _split_sentences(text):
    """Divide en oraciones sin cortar 'ARTÍCULO 12.', 'UNIDAD 1.', 'PROYECTO 2.' ni decimales como '3.0'."""
    protected = re.sub(
        r"\b(ART[ÍI]CULO|UNIDAD|PROYECTO)\s+(\d+)\.",
        r"\1 \2<DOT>",
        text,
        flags=re.IGNORECASE,
    )
    protected = re.sub(r"(\d)\.(\d)", r"\1<DEC>\2", protected)
    parts = [p.strip() for p in re.split(r"(?:\.\s+|\n+)", protected) if p.strip()]
    out = []
    for p in parts:
        s = p.replace("<DOT>", ".").replace("<DEC>", ".").strip()
        if not s.endswith("."):
            s += "."
        out.append(s)
    return out


def _best_sentence(question, contexts, min_overlap=0.18):
    """Selector de oración con soporte explícito para abstención."""
    q_words = [w for w in _tokenize_es(question) if w not in _STOPWORDS_ES]
    q_stems = {_stem_es(w) for w in q_words}
    if not contexts or not q_stems:
        return "No está en los documentos.", None, 0.0

    best_sent = None
    best_meta = None
    best_score = 0.0

    for doc_text, meta, sim in contexts:
        for sent in _split_sentences(doc_text):
            s_words = [w for w in _tokenize_es(sent) if w not in _STOPWORDS_ES]
            s_stems = {_stem_es(w) for w in s_words}
            if len(s_stems) < 2:
                continue
            overlap = len(q_stems & s_stems) / len(q_stems)
            exact = len(set(q_words) & set(s_words)) / max(1, len(q_words))
            score = 0.65 * overlap + 0.25 * exact + 0.10 * max(0.0, float(sim))
            if score > best_score:
                best_score = score
                best_sent = sent
                best_meta = meta

    if best_sent is None or best_score < min_overlap:
        return "No está en los documentos.", None, best_score

    cite = f"[{best_meta['fuente']}:chunk_{best_meta['chunk']}]" if best_meta else ""
    return f"{best_sent} {cite}".strip(), best_meta, best_score


def generate(question, contexts, model="llama3.2:1b", use_citations=True,
             temperature=0.0, force_mode=None):
    """Genera respuesta usando el contexto recuperado.

    - Si Ollama está activo (y force_mode != 'extractive'/'raw_chunk'), usa el LLM.
    - Si force_mode == 'raw_chunk', devuelve el primer chunk tal cual (reproduce baseline).
    - Si force_mode == 'no_rag', responde sin contexto para medir la ganancia del RAG.
    - En modo local por defecto usa selección de oración con verificación de soporte.
    """
    if force_mode == "no_rag":
        if ollama_available():
            import ollama
            r = ollama.generate(
                model=model,
                prompt=f"Responde de forma breve y exacta: {question}",
                options={"temperature": temperature},
            )
            return r["response"].strip(), f"ollama_no_rag:{model}"
        return ("No dispongo de los documentos internos de la UAO en memoria "
                "para verificar ese dato específico."), "no_rag_baseline"

    if force_mode == "raw_chunk":
        if not contexts:
            return "No está en los documentos.", "raw_chunk"
        return contexts[0][0][:500], "raw_chunk"

    norm_contexts = [
        (c[0], c[1], c[2] if len(c) > 2 else 0.0) for c in contexts
    ]

    ans, meta, _ = _best_sentence(question, norm_contexts)

    if force_mode != "extractive" and ollama_available():
        import ollama
        if meta is None:
            return "No está en los documentos.", f"ollama:{model}"
        sent_only = re.sub(r"\s*\[[^\]]+\]\s*$", "", ans).strip()
        cite = f"[{meta['fuente']}:chunk_{meta['chunk']}]" if use_citations else ""
        prompt = (
            "Eres un asistente académico de la Universidad Autónoma de Occidente (UAO).\n"
            "Responde la pregunta en español en una sola oración usando la evidencia recuperada "
            "y conservando literalmente las cifras, fechas y nombres técnicos.\n"
            f"Evidencia recuperada: {sent_only}\n"
            f"Pregunta: {question}\n"
            "Respuesta:"
        )
        try:
            r = ollama.generate(model=model, prompt=prompt, options={"temperature": temperature})
            llm_out = re.sub(r"\s*\[[^\]]+\]\s*", " ", r.get("response", "")).strip().replace("\n", " ")
        except Exception:
            llm_out = ""
        req_stems = {_stem_es(w) for w in _tokenize_es(sent_only) if w not in _STOPWORDS_ES}
        out_stems = {_stem_es(w) for w in _tokenize_es(llm_out) if w not in _STOPWORDS_ES}
        if (not llm_out or "no está" in llm_out.lower() or "no aparece" in llm_out.lower()
                or not req_stems.issubset(out_stems)):
            final_txt = sent_only
        else:
            final_txt = llm_out
        return (f"{final_txt} {cite}".strip() if cite else final_txt), f"ollama:{model}"

    if not use_citations and meta is not None:
        ans = re.sub(r"\s*\[[^\]]+\]\s*$", "", ans)
    return ans, "extractive_sentence"


# ---------- 4. Evaluación completa (Retrieval + RAGAS proxy + F1/EM) ----------

def _f1_tokens(pred, gold):
    p_toks = [w for w in _tokenize_es(pred) if w not in _STOPWORDS_ES]
    g_toks = [w for w in _tokenize_es(gold) if w not in _STOPWORDS_ES]
    if not p_toks or not g_toks:
        p_toks = _tokenize_es(pred)
        g_toks = _tokenize_es(gold)
    if not p_toks or not g_toks:
        return 0.0
    p_stems = Counter(_stem_es(w) for w in p_toks)
    g_stems = Counter(_stem_es(w) for w in g_toks)
    common = sum((p_stems & g_stems).values())
    if common == 0:
        return 0.0
    prec = common / sum(p_stems.values())
    rec = common / sum(g_stems.values())
    return 2 * prec * rec / (prec + rec)


def evaluate(gold_qa, store, embed_fn, k=3, model="llama3.2:1b",
             use_citations=True, force_mode=None):
    """Evalúa un conjunto de preguntas {q, a_esperada, fuente}.

    Devuelve métricas agregadas y el detalle fila por fila.
    """
    hits_1 = 0
    hits_k = 0
    mrr = 0.0
    ctx_prec_list = []
    ctx_rec_list = []
    faithful_list = []
    ans_rel_list = []
    f1_list = []
    em_list = []
    details = []

    for item in gold_qa:
        q = item["q"]
        expected = item["a_esperada"]
        exp_source = item["fuente"]

        q_emb = embed_fn([q])[0]
        res = store.query(q_emb, k=max(k, 3))
        res_k = res[:k]
        docs = [r[0] for r in res_k]
        metas = [r[1] for r in res_k]

        # Hit@1, Hit@k y MRR verificando además que el chunk contenga la respuesta
        exp_norm = expected.lower().strip()
        ranks = []
        rel_flags = []
        for idx, (doc_text, meta) in enumerate(zip(docs, metas), start=1):
            same_source = exp_source in str(meta.get("fuente", ""))
            has_answer = exp_norm in doc_text.lower()
            is_rel = same_source and (has_answer or len(docs) == 1)
            if not is_rel and same_source:
                # Si la respuesta está parafraseada, verificamos solapamiento de términos
                g_stems = {_stem_es(w) for w in _tokenize_es(expected)}
                d_stems = {_stem_es(w) for w in _tokenize_es(doc_text)}
                is_rel = len(g_stems & d_stems) >= max(1, int(0.6 * len(g_stems)))
            rel_flags.append(1 if is_rel else 0)
            if is_rel:
                ranks.append(idx)

        rank = ranks[0] if ranks else None
        if rank == 1:
            hits_1 += 1
        if rank is not None and rank <= k:
            hits_k += 1
            mrr += 1.0 / rank

        # Context Precision@k (precisión ponderada por posición, estilo RAGAS)
        if any(rel_flags):
            prec_at_i = []
            running = 0
            for idx, flag in enumerate(rel_flags, start=1):
                if flag:
                    running += 1
                    prec_at_i.append(running / idx)
            ctx_prec = sum(prec_at_i) / len(prec_at_i)
        else:
            ctx_prec = 0.0
        ctx_prec_list.append(ctx_prec)

        # Context Recall: fracción de palabras clave de la respuesta esperada en el top-k
        joined_ctx = " ".join(docs).lower()
        exp_words = [w for w in _tokenize_es(expected) if w not in _STOPWORDS_ES] or _tokenize_es(expected)
        ctx_words = set(_tokenize_es(joined_ctx))
        ctx_stems = {_stem_es(w) for w in ctx_words}
        ctx_rec = sum(_stem_es(w) in ctx_stems for w in exp_words) / max(1, len(exp_words))
        ctx_rec_list.append(ctx_rec)

        # Generación
        ans, backend_used = generate(
            q, res_k, model=model, use_citations=use_citations, force_mode=force_mode
        )
        ans_clean = re.sub(r"\[[^\]]+\]", "", ans).strip()

        # Faithfulness: fracción de tokens con contenido de la respuesta apoyados por el contexto
        ans_words = [w for w in _tokenize_es(ans_clean) if w not in _STOPWORDS_ES] or _tokenize_es(ans_clean)
        if force_mode == "no_rag":
            faith = 0.0
        else:
            faith = sum(_stem_es(w) in ctx_stems for w in ans_words) / max(1, len(ans_words))
        faithful_list.append(faith)

        # Answer Relevance: solapamiento léxico/morfólogico con la pregunta + presencia de respuesta
        q_content = {_stem_es(w) for w in _tokenize_es(q) if w not in _STOPWORDS_ES}
        a_content = {_stem_es(w) for w in ans_words}
        if "no está en los documentos" in ans_clean.lower():
            ans_rel = 0.0
        else:
            q_overlap = len(q_content & a_content) / max(1, len(q_content))
            ans_rel = min(1.0, 0.5 * q_overlap + 0.5 * min(1.0, len(a_content) / 4.0))
        ans_rel_list.append(ans_rel)

        f1 = _f1_tokens(ans_clean, expected)
        em = 1.0 if exp_norm in ans_clean.lower() else 0.0
        f1_list.append(f1)
        em_list.append(em)

        details.append({
            "q": q,
            "a_esperada": expected,
            "fuente_esperada": exp_source,
            "respuesta": ans,
            "rank": rank,
            "hit@1": 1 if rank == 1 else 0,
            "hit@k": 1 if (rank is not None and rank <= k) else 0,
            "faithfulness": round(faith, 3),
            "answer_relevance": round(ans_rel, 3),
            "context_precision": round(ctx_prec, 3),
            "context_recall": round(ctx_rec, 3),
            "F1": round(f1, 3),
            "EM": int(em),
            "backend": backend_used,
        })

    n = max(1, len(gold_qa))
    summary = {
        "hit_rate@1": round(hits_1 / n, 4),
        "hit_rate@k": round(hits_k / n, 4),
        "MRR": round(mrr / n, 4),
        "context_precision": round(sum(ctx_prec_list) / n, 4),
        "context_recall": round(sum(ctx_rec_list) / n, 4),
        "faithfulness_proxy": round(sum(faithful_list) / n, 4),
        "answer_relevance": round(sum(ans_rel_list) / n, 4),
        "F1": round(sum(f1_list) / n, 4),
        "EM": round(sum(em_list) / n, 4),
        "k": k,
        "n": len(gold_qa),
    }
    return summary, details


def build_index_from_dir(doc_dir="rag_docs", chunk_size=120, overlap=20,
                         embedder="hybrid", persist_dir=None, prefer_chroma=True):
    """Construye el índice sobre los documentos de doc_dir y devuelve (store, embed_fn, chunks_info)."""
    files = sorted(
        f for f in os.listdir(doc_dir)
        if f.startswith("doc") and f.endswith((".txt", ".md"))
    )
    all_chunks, all_metas = [], []
    for fname in files:
        full_path = os.path.join(doc_dir, fname)
        pieces = chunk(load_doc(full_path), size=chunk_size, overlap=overlap)
        for j, piece in enumerate(pieces):
            all_chunks.append(piece)
            all_metas.append({"fuente": fname, "chunk": j, "n_words": len(piece.split())})

    if embedder == "ollama" and ollama_available():
        embed_fn = OllamaHybridEmbedder().fit(all_chunks)
    elif embedder == "tfidf":
        embed_fn = TfidfEmbedder().fit(all_chunks)
    else:
        embed_fn = HybridDenseEmbedder().fit(all_chunks)

    store = Store(persist_dir=persist_dir, prefer_chroma=prefer_chroma)
    embs = embed_fn(all_chunks)
    ids = [f"{m['fuente']}#{m['chunk']}" for m in all_metas]
    store.add(ids, all_chunks, embs, all_metas)
    return store, embed_fn, all_chunks, all_metas


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    rag_dir = os.path.join(here, "rag_docs")
    gold_path = os.path.join(rag_dir, "gold_qa.json")
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    # 1. Reproducir baseline TF-IDF + chunk crudo
    st_tfidf, fn_tfidf, chunks_b, _ = build_index_from_dir(
        rag_dir, chunk_size=120, overlap=20, embedder="tfidf", prefer_chroma=False
    )
    b_k1, _ = evaluate(gold, st_tfidf, fn_tfidf, k=1, force_mode="raw_chunk")
    b_k3, _ = evaluate(gold, st_tfidf, fn_tfidf, k=3, force_mode="raw_chunk")
    print("Baseline TF-IDF k=1:", json.dumps(b_k1, ensure_ascii=False))
    print("Baseline TF-IDF k=3:", json.dumps(b_k3, ensure_ascii=False))

    # 2. Pipeline mejorado con embeddings densos híbridos + respuesta con citas
    st_hyb, fn_hyb, chunks_h, _ = build_index_from_dir(
        rag_dir, chunk_size=120, overlap=20, embedder="hybrid", prefer_chroma=False
    )
    h_k3, details = evaluate(gold, st_hyb, fn_hyb, k=3, force_mode="extractive")
    print("Híbrido denso k=3:", json.dumps(h_k3, ensure_ascii=False))
    assert h_k3["hit_rate@k"] >= 0.85, "retrieval bajo en evaluación principal"
    assert h_k3["EM"] >= 0.70, "EM bajo en evaluación principal"
    print("SELF-CHECK OK")

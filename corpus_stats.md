# corpus_stats.md - Corpus baseline vs nuevo (congelado Día 1)

| Métrica | Shakespeare (baseline notebook) | TinyStories-valid (corpus nuevo) |
|---|---|---|
| Archivo | `shakespeare_baseline.txt` | `corpus_nuevo.txt` |
| Caracteres | 1,115,394 | 19,432,979 |
| Palabras | 202,651 | 3,775,640 |
| Vocab char (lower) | 39 | 72 |
| Vocab word | 23,641 | 38,486 |
| Fuente | Karpathy char-rnn / TF | `roneneldan/TinyStories` valid (Eldan & Li 2023) |

## Decisión para entrenamiento (comparativa limpia)

- Usar subset comparable: **primer 1,000,000 caracteres de `corpus_nuevo.txt`** (~200K palabras, mismo orden de magnitud que Shakespeare).
- Split idéntico al notebook: train `[:1M]`, val `[1M:1.06M]` (o 90/10 si el subset es menor), test resto.
- Char: `CharTok` → vocab ~70. Word: `WordTok(max_tokens=10000)` → medir %OOV en val.
- Por qué TinyStories: vocabulario simple inglés 3-4 años, modelos <10M coherentes (paper), converge en Colab T4. Cita: `estado_arte/BLOQUE_B_llama_corpus_metrica_char/Architectural_Tradeoffs_Small_LMs_Bhatti_2025.md`.
- Alternativa diferencial (si el grupo quiere español): reemplazar por Quijote+Wikitext-ES y regenerar esta tabla.

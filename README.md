# Proyecto 2 — Generación de texto con Transformers tipo LLaMA y RAG en Ollama

Proyecto de la asignatura **Inteligencia Artificial Generativa** (Universidad Autónoma de Occidente). Tiene tres partes:

1. Generación a nivel de carácter con una arquitectura decoder-only tipo LLaMA.
2. Generación a nivel de palabra sobre el mismo corpus y comparación rigurosa contra el nivel de carácter.
3. Sistema RAG (Retrieval-Augmented Generation) con Ollama para responder preguntas sobre documentos académicos de la UAO.

## Resultados principales

Resultados generados en CPU con la configuración reducida. En GPU T4 (Kaggle o Colab) los notebooks cambian automáticamente a `d=256`, 6 capas, 8 cabezas y 3000/2500 pasos.

| Modelo | Vocabulario | Parámetros | Test loss | Test PPL | Test BPC | Test acc | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Char-LLaMA | 59 | 0.34M | 1.4194 | 4.13 | 2.048 | 0.567 | 0.175 |
| Word-LLaMA | 10,000 | 1.29M | 5.7682 | 319.97 | 1.603 | 0.110 | 0.143 |

La perplejidad no se compara directamente entre los dos modelos porque el espacio de salida es distinto (59 caracteres frente a 10,000 palabras). La comparación justa es el **BPC normalizado por carácter**: el modelo de palabra necesita 1.603 bits por carácter y el de carácter 2.048.

| Sistema RAG | Hit-rate@1 | MRR | Faithfulness | Exact Match |
|---|---:|---:|---:|---:|
| Sin RAG | 0.00 | 0.00 | 0.00 | 0.00 |
| TF-IDF, top-k=1 | 0.75 | 0.75 | 0.99 | 0.50 |
| TF-IDF, top-k=3 | 0.75 | 0.875 | 0.99 | 0.50 |
| Denso + citas, top-k=3 | 1.00 | 1.00 | 1.00 | 0.95 |

## Contenido del repositorio

| Archivo o carpeta | Qué contiene |
|---|---|
| `01_char_llama.ipynb` | Punto 1 completo: teoría, modelo, entrenamiento, curvas, muestreo y análisis. |
| `02_word_llama.ipynb` | Punto 2 completo: curva OOV, entrenamiento word-level y comparativa contra carácter. |
| `03_rag_ollama.ipynb` | Punto 3 completo: ingesta, chunking, índice, evaluación, ablación y demo. |
| `llama_mini.py` | Modelo tipo LLaMA, tokenizadores, entrenamiento, checkpoints y muestreo. |
| `rag_local.py` | Pipeline RAG, backends de embeddings, generación con citas y métricas. |
| `rag_docs/` | Tres documentos UAO y las 20 preguntas oro. |
| `figuras/` | Curvas y comparativas exportadas por los notebooks. |
| `checkpoints/` | Pesos de los modelos y métricas en JSON. |
| `corpus_stats.md` | Comparación del corpus nuevo contra Shakespeare y decisión del subset. |

## Cómo ejecutar

### Kaggle o Google Colab

1. Subir el repositorio o los tres notebooks.
2. Activar una GPU T4.
3. Ejecutar en orden `01_char_llama.ipynb`, `02_word_llama.ipynb` y `03_rag_ollama.ipynb`.

Los notebooks buscan `corpus_nuevo.txt` en el directorio local y en `/kaggle/input`; si no lo encuentran, descargan TinyStories desde Hugging Face. El notebook 3 instala Ollama, ChromaDB y los modelos `nomic-embed-text` y `llama3.2:1b` cuando corre en Linux.

### Local

```bash
python llama_mini.py      # self-check del modelo
python rag_local.py       # self-check del RAG
jupyter notebook
```

`llama_mini.py` solo requiere `torch`. `rag_local.py` funciona sin Ollama usando el backend denso local.

## Decisiones de diseño

- **Corpus:** TinyStories en inglés. Se usa el primer millón de caracteres para entrenamiento, los siguientes 60,000 para validación y el resto para prueba, con los cortes ajustados al siguiente espacio en blanco.
- **Arquitectura:** pre-RMSNorm, RoPE con base 10,000, SwiGLU sin bias y weight tying. Sin GQA ni caché KV por el alcance del proyecto.
- **Tokenizadores:** `CharTok` reserva el id 0 para `<UNK>`. `WordTok` se construye solo con el texto de entrenamiento para evitar fuga de datos y mide OOV en validación y prueba.
- **Métricas:** loss, perplejidad, accuracy, BPC normalizado, Distinct-1/2, repetición de trigramas, BLEU-2 y ROUGE-L.
- **RAG:** chunking de 120 palabras con overlap de 20, recuperación top-k, respuesta con citas `[documento:chunk]` y abstención cuando la respuesta no está en el contexto.

## Limitaciones

- Los resultados incluidos se obtuvieron en CPU con una configuración reducida. Para la entrega final conviene ejecutar los notebooks en T4 y actualizar las tablas.
- Los documentos del RAG son extractos cortos: generan solo seis fragmentos y la recuperación resulta relativamente fácil. Un corpus más largo exigiría re-ranking.
- El modelo no usa GQA ni caché KV. Son mejoras posibles con más cómputo, no errores de implementación.
- El backend local del RAG es una aproximación determinista. Las métricas definitivas deben calcularse con `nomic-embed-text` y `llama3.2:1b` en Ollama.

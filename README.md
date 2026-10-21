# Proyecto 2 - Generación de texto con Transformers tipo LLaMA y RAG en Ollama

Proyecto de la asignatura **Inteligencia Artificial Generativa** (Universidad Autónoma de Occidente). Tiene tres partes:

**Estudiantes:** Samuel Patiño Lucumí y Nicolás Peña Irurita

1. Generación a nivel de carácter con una arquitectura decoder-only tipo LLaMA.
2. Generación a nivel de palabra sobre el mismo corpus y comparación rigurosa contra el nivel de carácter.
3. Sistema RAG (Retrieval-Augmented Generation) con Ollama para responder preguntas sobre documentos académicos de la UAO.

## Resultados principales

Los tres puntos se ejecutaron en Tesla T4. Los puntos 1 y 2 usan la configuración completa. El punto 3 usa `nomic-embed-text` y `llama3.2:1b` en Ollama.

| Modelo | Vocabulario | Parámetros | Test loss | Test PPL | Test BPC | Test acc | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Char-LLaMA | 59 | 4.73M | 0.9085 | 2.48 | 1.311 | 0.726 | 0.158 |
| Word-LLaMA | 10,000 | 7.28M | 4.2771 | 72.0 | 1.193 | 0.285 | 0.145 |

La perplejidad no se compara directamente entre los dos modelos porque el espacio de salida es distinto (59 caracteres frente a 10,000 palabras). La comparación justa es el **BPC normalizado por carácter**: el modelo de palabra necesita 1.193 bits por carácter y el de carácter 1.311.

| Sistema RAG | Hit-rate@1 | MRR | Faithfulness | Exact Match |
|---|---:|---:|---:|---:|
| Sin RAG | 0.00 | 0.00 | 0.00 | 0.00 |
| TF-IDF, top-k=1 | 0.75 | 0.75 | 0.99 | 0.50 |
| TF-IDF, top-k=3 | 0.75 | 0.875 | 0.99 | 0.50 |
| Denso + citas, top-k=3 | 0.85 | 0.8667 | 0.60 | 0.35 |

La tabla corresponde a la corrida con `nomic-embed-text` y `llama3.2:1b` en Ollama. La recuperación es sólida: el fragmento correcto aparece en el top-3 en la mayoría de las preguntas. La generación es conservadora: el modelo se abstiene en varias preguntas aunque el contexto contenga la respuesta. La versión corregida de `rag_local.py` añade una verificación de evidencia que devuelve la frase textual con su cita cuando el modelo no conserva los datos; esa mejora queda documentada en el código y en la sección de limitaciones.

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
| `requirements.txt` | Dependencias mínimas para ejecutar los notebooks. |
| `CREDITOS.md` | Papers, repositorios y documentación usados como referencia. |
| `corpus_stats.md` | Comparación del corpus nuevo contra Shakespeare y decisión del subset. |

## Cómo ejecutar

### Kaggle o Google Colab

1. Crear un notebook con GPU T4 y activar el acceso a internet.
2. Subir el archivo `.ipynb` o clonar el repositorio completo.
3. Ejecutar en orden `01_char_llama.ipynb`, `02_word_llama.ipynb` y `03_rag_ollama.ipynb`.

La primera celda de cada notebook descarga `llama_mini.py`, `rag_local.py`, `gold_qa.json` o `char_metrics.json` desde el repositorio público si no están en el entorno. Así el notebook funciona aunque se suba solo el archivo `.ipynb`. El notebook 3 instala Ollama, ChromaDB y los modelos `nomic-embed-text` y `llama3.2:1b` cuando corre en Linux.

### Local

```bash
pip install -r requirements.txt
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

- **Contexto corto:** la ventana es de 96 a 128 caracteres o de 48 a 64 palabras. La coherencia se mantiene a nivel local; para textos largos haría falta más contexto.
- **OOV en el modelo de palabra:** con un vocabulario de 10,000 palabras, el 3.1% de los tokens de validación quedan como `<UNK>`. Nombres propios y palabras raras se pierden.
- **RAG con documentos cortos:** los tres extractos generan solo seis fragmentos. La recuperación es relativamente fácil y las métricas altas deben leerse con esa salvedad. Un corpus real exigiría más documentos y un paso de re-ranking.
- **Métricas de RAG aproximadas:** `faithfulness` y `answer_relevance` se calculan con solapamiento léxico y morfológico, no con la suite RAGAS completa y un LLM juez. RAGAS queda como extensión.
- **Generación con `llama3.2:1b`:** el modelo se abstiene en varias preguntas aunque el contexto contenga la respuesta. La verificación de evidencia reduce el problema, pero no lo elimina.
- **Sin GQA ni caché KV:** no son necesarios para este tamaño de modelo, pero limitan escalar a contextos largos o a modelos mayores.

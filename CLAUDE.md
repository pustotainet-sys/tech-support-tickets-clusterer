# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the TF-IDF based clustering script
pipenv run python analyze_support_requests.py

# Run the embeddings-based clustering script (preferred)
pipenv run python analyze_embeddings.py

# Install dependencies
pipenv install
```

## Environment Setup

Copy `.env.example` to `.env` and fill in the required API keys:

```
CLOUDRU_API_KEY=...        # Required for embeddings (analyze_embeddings.py)
YANDEX_API_KEY=...         # Optional: enables LLM cluster naming
YANDEX_FOLDER_ID=...       # Required if YANDEX_API_KEY is set
```

## Architecture

There are two independent analysis scripts, each producing Excel output:

### `analyze_embeddings.py` (primary, self-contained)
Pipeline: load Excel → clean text → Cloud.ru Embeddings (`BAAI/bge-m3`, 1024-dim) → UMAP dimensionality reduction → KMeans → YandexGPT cluster naming → Excel output.

- Embeddings are cached to `embeddings_cache.npy` — delete this file to force recomputation.
- If `YANDEX_API_KEY` is not set, clusters are named by number only.
- Cloud.ru API is accessed via the OpenAI-compatible client pointed at `https://foundation-models.api.cloud.ru/v1`.
- YandexGPT is accessed via `https://llm.api.cloud.yandex.net/v1` with model URI format `gpt://{folder_id}/yandexgpt-lite`.

### `analyze_support_requests.py` (legacy, has external dependency)
Pipeline: load Excel → clean text → TF-IDF vectorization → KMeans → YandexGPT naming via the `kaiten-chat-rag-2` project's internal LLM wrapper.

- **Requires** the sibling project at `C:\Users\VivoBook 17X\Kaiten Code\kaiten-chat-rag-2` and loads its `.env`.
- Uses `sklearn` TF-IDF + KMeans with Russian stop words; no embeddings.

### Input/Output
- Input: `DataExport_support22032026.xlsx` — columns `request_description` (primary) and `card_title` (fallback when description is empty).
- Output per script:
  - `support_clusters_embeddings.xlsx` / `support_clusters_result.xlsx` — all rows with `cluster_id` and `cluster_name` columns added.
  - `cluster_statistics_embeddings.xlsx` / `cluster_statistics.xlsx` — per-cluster count and percentage summary.

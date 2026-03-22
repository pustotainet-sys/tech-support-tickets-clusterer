"""
Скрипт кластеризации запросов техподдержки с использованием Cloud.ru Embeddings + KMeans.

Полностью автономный скрипт без зависимостей от других проектов.

Требуемые переменные окружения в .env:
- CLOUDRU_API_KEY: API ключ Cloud.ru для embeddings
- YANDEX_API_KEY: API ключ Yandex Cloud для YandexGPT
- YANDEX_FOLDER_ID: ID папки в Yandex Cloud
"""

import os
import re
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from umap import UMAP
from sklearn.cluster import KMeans


# Загружаем .env из текущей директории
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / '.env')


# ============== КОНФИГУРАЦИЯ ==============

# Файлы
INPUT_FILE = SCRIPT_DIR / "DataExport_support22032026.xlsx"
OUTPUT_FILE = SCRIPT_DIR / "support_clusters_embeddings.xlsx"
STATS_FILE = SCRIPT_DIR / "cluster_statistics_embeddings.xlsx"
CACHE_FILE = SCRIPT_DIR / "embeddings_cache.npy"

# Cloud.ru Embeddings
CLOUDRU_API_KEY = os.getenv('CLOUDRU_API_KEY')
CLOUDRU_API_BASE = "https://foundation-models.api.cloud.ru/v1"
EMBEDDING_MODEL = 'BAAI/bge-m3'
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_DIMENSIONS = 1024

# YandexGPT
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_BASE = "https://llm.api.cloud.yandex.net/v1"
YANDEX_MODEL = "yandexgpt-lite"

# KMeans параметры
N_CLUSTERS = 10

# UMAP параметры
UMAP_N_COMPONENTS = 50
UMAP_N_NEIGHBORS = 15


# ============== EMBEDDING CLIENT ==============

class CloudRuEmbeddingClient:
    """Клиент для получения embeddings через Cloud.ru API (OpenAI-совместимый)."""

    def __init__(
        self,
        api_key: str,
        api_base: str = CLOUDRU_API_BASE,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ):
        self.model = model
        self.dimensions = dimensions
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получение embeddings для списка текстов."""
        # Заменяем переносы строк на пробелы
        texts = [t.replace('\n', ' ') for t in texts]

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )

        return [item.embedding for item in response.data]


# ============== LLM CLIENT ==============

class YandexGPTClient:
    """Клиент для YandexGPT (OpenAI-совместимый API)."""

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        api_base: str = YANDEX_API_BASE,
        model: str = YANDEX_MODEL,
        temperature: float = 0.3,
    ):
        self.folder_id = folder_id
        self.model = f"gpt://{folder_id}/{model}"
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def complete(self, prompt: str) -> str:
        """Генерация текста по промпту."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()


# ============== УТИЛИТЫ ==============

def clean_text(text: str) -> str:
    """Очистка текста от HTML-тегов и нормализация."""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = re.sub(r'<[^>]+>', ' ', text)      # HTML теги
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # HTML entities
    text = re.sub(r'http[s]?://\S+', ' ', text)  # URLs
    text = re.sub(r'\S+@\S+', ' ', text)      # email
    text = re.sub(r'\s+', ' ', text)          # лишние пробелы
    return text.strip()


def load_data(file_path: Path) -> pd.DataFrame:
    """Загрузка данных из Excel."""
    print(f"Загрузка данных из {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Загружено {len(df)} записей")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка данных."""
    print("Предобработка текстов...")

    df['cleaned_text'] = df['request_description'].apply(clean_text)

    # Заменяем пустые на заголовок карточки
    mask = df['cleaned_text'] == ""
    df.loc[mask, 'cleaned_text'] = df.loc[mask, 'card_title'].apply(
        lambda x: clean_text(str(x)) if pd.notna(x) else ""
    )

    empty_count = (df['cleaned_text'] == "").sum()
    print(f"Пустых текстов после обработки: {empty_count}")
    return df


# ============== EMBEDDINGS ==============

def get_embeddings(
    texts: List[str],
    client: CloudRuEmbeddingClient,
    batch_size: int = EMBEDDING_BATCH_SIZE
) -> np.ndarray:
    """Получение embeddings батчами с прогресс-баром."""
    print(f"Получение embeddings для {len(texts)} текстов...")

    all_embeddings = []
    max_retries = 3

    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [t if t.strip() else "пустой запрос" for t in batch_texts]

        for attempt in range(max_retries):
            try:
                batch_embeddings = client.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"\nОшибка: {e}. Повтор через {wait_time}с...")
                    time.sleep(wait_time)
                else:
                    print(f"\nОшибка после {max_retries} попыток: {e}")
                    all_embeddings.extend([[0.0] * EMBEDDING_DIMENSIONS] * len(batch_texts))

    return np.array(all_embeddings)


def load_or_compute_embeddings(
    texts: List[str],
    client: CloudRuEmbeddingClient,
    cache_file: Path
) -> np.ndarray:
    """Загрузка embeddings из кэша или вычисление."""
    if cache_file.exists():
        print(f"Загрузка embeddings из кэша: {cache_file}")
        embeddings = np.load(cache_file)
        if len(embeddings) == len(texts):
            print(f"Загружено {len(embeddings)} embeddings из кэша")
            return embeddings
        print(f"Размер кэша не совпадает, пересчитываем...")

    embeddings = get_embeddings(texts, client)

    print(f"Сохранение embeddings в кэш: {cache_file}")
    np.save(cache_file, embeddings)

    return embeddings


# ============== КЛАСТЕРИЗАЦИЯ ==============

def cluster_with_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = N_CLUSTERS
) -> np.ndarray:
    """UMAP снижение размерности + KMeans кластеризация."""

    print(f"UMAP снижение размерности: {embeddings.shape[1]} -> {UMAP_N_COMPONENTS}...")
    reducer = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"UMAP завершен. Размерность: {reduced.shape}")

    print(f"KMeans кластеризация (n_clusters={n_clusters})...")
    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    cluster_labels = clusterer.fit_predict(reduced)

    print(f"Создано кластеров: {n_clusters}")
    print("Все записи распределены по кластерам (без шума)")

    return cluster_labels


# ============== ИМЕНОВАНИЕ КЛАСТЕРОВ ==============

def get_representative_samples(df: pd.DataFrame, cluster_id: int, n_samples: int = 10) -> List[str]:
    """Получение репрезентативных примеров для кластера."""
    cluster_df = df[df['cluster_id'] == cluster_id]

    if len(cluster_df) == 0:
        return []

    sample_df = cluster_df.sample(n=min(n_samples, len(cluster_df)), random_state=42)

    samples = []
    for _, row in sample_df.iterrows():
        text = row['request_description']
        if pd.notna(text) and str(text).strip():
            samples.append(str(text)[:500])

    return samples


def name_cluster_with_llm(client: YandexGPTClient, samples: List[str]) -> str:
    """Генерация названия кластера через YandexGPT."""
    if not samples:
        return "Кластер (без примеров)"

    samples_text = "\n---\n".join(samples[:7])

    prompt = f"""Ты аналитик техподдержки. Проанализируй примеры обращений пользователей и определи общую тему/домен.

Примеры обращений:
{samples_text}

Напиши ТОЛЬКО краткое название домена/темы (2-5 слов), которое объединяет эти обращения.
Название должно быть на русском языке, без пояснений.

Название домена:"""

    try:
        name = client.complete(prompt)
        name = re.sub(r'^["\'\-\*]+|["\'\-\*]+$', '', name)
        return name.strip() if name.strip() else "Кластер (ошибка)"
    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return f"Кластер (ошибка: {str(e)[:30]})"


def name_all_clusters(df: pd.DataFrame, llm_client: Optional[YandexGPTClient]) -> dict:
    """Именование всех кластеров."""
    cluster_ids = sorted([c for c in df['cluster_id'].unique() if c != -1])
    cluster_names = {}

    if llm_client:
        print("Именование кластеров через YandexGPT...")
        for cluster_id in tqdm(cluster_ids, desc="Именование кластеров"):
            samples = get_representative_samples(df, cluster_id)
            name = name_cluster_with_llm(llm_client, samples)
            cluster_names[cluster_id] = name
            print(f"  Кластер {cluster_id}: {name}")
    else:
        print("YandexGPT не настроен, используем номера кластеров...")
        for cluster_id in cluster_ids:
            count = (df['cluster_id'] == cluster_id).sum()
            cluster_names[cluster_id] = f"Кластер {cluster_id} ({count} записей)"

    cluster_names[-1] = "Без кластера (шум)"
    return cluster_names


# ============== СОХРАНЕНИЕ ==============

def save_results(
    df: pd.DataFrame,
    cluster_names: dict,
    output_file: Path,
    stats_file: Path
):
    """Сохранение результатов в Excel."""
    print("Сохранение результатов...")

    df['cluster_name'] = df['cluster_id'].map(cluster_names)

    output_df = df.drop(columns=['cleaned_text'], errors='ignore')
    output_df.to_excel(output_file, index=False)
    print(f"Результаты сохранены в {output_file}")

    stats_data = []
    for cluster_id in sorted(cluster_names.keys()):
        count = (df['cluster_id'] == cluster_id).sum()
        if count > 0:
            stats_data.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_names[cluster_id],
                'count': count,
                'percentage': round(count / len(df) * 100, 2)
            })

    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('count', ascending=False)
    stats_df.to_excel(stats_file, index=False)
    print(f"Статистика сохранена в {stats_file}")


# ============== MAIN ==============

def main():
    """Основная функция."""
    print("=" * 70)
    print("Кластеризация запросов техподдержки")
    print("Метод: Cloud.ru Embeddings + UMAP + KMeans")
    print("=" * 70)

    # Проверка конфигурации
    if not CLOUDRU_API_KEY:
        print("ОШИБКА: Не задан CLOUDRU_API_KEY в .env")
        print("Создайте файл .env с переменными:")
        print("  CLOUDRU_API_KEY=ваш_ключ")
        print("  YANDEX_API_KEY=ваш_ключ (опционально)")
        print("  YANDEX_FOLDER_ID=ваш_folder_id (опционально)")
        return

    # 1. Загрузка данных
    df = load_data(INPUT_FILE)

    # 2. Предобработка
    df = preprocess_data(df)

    # 3. Инициализация Cloud.ru Embeddings
    print("Инициализация Cloud.ru Embeddings...")
    embed_client = CloudRuEmbeddingClient(
        api_key=CLOUDRU_API_KEY,
        model=EMBEDDING_MODEL,
    )
    print(f"Модель: {EMBEDDING_MODEL} (dim={EMBEDDING_DIMENSIONS})")

    # 4. Получение embeddings (с кэшированием)
    texts = df['cleaned_text'].tolist()
    embeddings = load_or_compute_embeddings(texts, embed_client, CACHE_FILE)

    # 5. Кластеризация
    cluster_labels = cluster_with_kmeans(embeddings)
    df['cluster_id'] = cluster_labels

    # 6. Именование кластеров
    llm_client = None
    if YANDEX_API_KEY and YANDEX_FOLDER_ID:
        print("Инициализация YandexGPT...")
        llm_client = YandexGPTClient(
            api_key=YANDEX_API_KEY,
            folder_id=YANDEX_FOLDER_ID,
        )

    cluster_names = name_all_clusters(df, llm_client)

    # 7. Сохранение
    save_results(df, cluster_names, OUTPUT_FILE, STATS_FILE)

    print("=" * 70)
    print("Анализ завершен!")
    print(f"Результаты: {OUTPUT_FILE}")
    print(f"Статистика: {STATS_FILE}")
    print(f"Кэш embeddings: {CACHE_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()

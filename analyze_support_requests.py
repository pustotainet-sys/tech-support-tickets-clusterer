"""
Скрипт для анализа и кластеризации запросов в техподдержку.

Использует TF-IDF + K-Means для кластеризации и YandexGPT для именования кластеров.
"""

import sys
import re
import os
from pathlib import Path

# Добавляем путь к основному проекту для импорта YandexGPT
MAIN_PROJECT_PATH = r'C:\Users\VivoBook 17X\Kaiten Code\kaiten-chat-rag-2'
sys.path.insert(0, MAIN_PROJECT_PATH)

# Загружаем .env из основного проекта
from dotenv import load_dotenv
load_dotenv(Path(MAIN_PROJECT_PATH) / '.env')

# Меняем рабочую директорию для корректной загрузки конфига
os.chdir(MAIN_PROJECT_PATH)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

# Импорт YandexGPT из основного проекта
from src.utils.config import SingleConfig
from src.utils.llm_embed_fabric.llms.yandexgpt import create_llm


# Конфигурация
INPUT_FILE = Path(__file__).parent / "DataExport_support22032026.xlsx"
OUTPUT_FILE = Path(__file__).parent / "support_clusters_result.xlsx"
STATS_FILE = Path(__file__).parent / "cluster_statistics.xlsx"
N_CLUSTERS = 12
SAMPLES_FOR_NAMING = 10


def clean_text(text: str) -> str:
    """Очистка текста от HTML-тегов и нормализация."""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Удаляем HTML теги
    text = re.sub(r'<[^>]+>', ' ', text)
    # Удаляем HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Удаляем URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    # Удаляем email адреса
    text = re.sub(r'\S+@\S+', ' ', text)
    # Удаляем лишние пробелы и переносы
    text = re.sub(r'\s+', ' ', text)
    # Убираем пробелы по краям
    text = text.strip()

    return text


def load_data(file_path: Path) -> pd.DataFrame:
    """Загрузка данных из Excel."""
    print(f"Загрузка данных из {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Загружено {len(df)} записей")
    print(f"Колонки: {list(df.columns)}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка данных."""
    print("Предобработка текстов...")

    # Очистка текста request_description
    df['cleaned_text'] = df['request_description'].apply(clean_text)

    # Фильтруем пустые тексты
    empty_count = (df['cleaned_text'] == "").sum()
    print(f"Пустых текстов: {empty_count}")

    # Заменяем пустые на заголовок карточки
    mask = df['cleaned_text'] == ""
    df.loc[mask, 'cleaned_text'] = df.loc[mask, 'card_title'].apply(
        lambda x: clean_text(str(x)) if pd.notna(x) else ""
    )

    return df


def vectorize_and_cluster(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> tuple:
    """TF-IDF векторизация и K-Means кластеризация."""
    print(f"Векторизация текстов (TF-IDF)...")

    # Русские стоп-слова
    russian_stop_words = [
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
        'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
        'только', 'её', 'мне', 'было', 'вот', 'от', 'меня', 'ещё', 'нет', 'о', 'из',
        'ему', 'теперь', 'когда', 'уже', 'вам', 'ни', 'очень', 'об', 'для', 'это',
        'этот', 'эта', 'эти', 'этим', 'этой', 'этого', 'есть', 'был', 'была', 'были',
        'быть', 'будет', 'если', 'при', 'до', 'можно', 'нужно', 'надо', 'также',
        'здравствуйте', 'добрый', 'день', 'привет', 'пожалуйста', 'спасибо'
    ]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words=russian_stop_words
    )

    # Фильтруем записи с пустым текстом
    valid_mask = df['cleaned_text'] != ""
    texts = df.loc[valid_mask, 'cleaned_text'].tolist()

    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"Размер TF-IDF матрицы: {tfidf_matrix.shape}")

    print(f"Кластеризация K-Means (n_clusters={n_clusters})...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )

    # Кластеризация только для валидных записей
    cluster_labels = np.full(len(df), -1)
    cluster_labels[valid_mask] = kmeans.fit_predict(tfidf_matrix)

    df['cluster_id'] = cluster_labels

    print("Кластеризация завершена")

    return df, vectorizer, kmeans, tfidf_matrix, valid_mask


def get_top_terms_per_cluster(
    vectorizer: TfidfVectorizer,
    kmeans: KMeans,
    n_terms: int = 10
) -> dict:
    """Получение топ-N термов для каждого кластера."""
    feature_names = vectorizer.get_feature_names_out()
    top_terms = {}

    for cluster_id in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-n_terms:][::-1]
        top_terms[cluster_id] = [feature_names[i] for i in top_indices]

    return top_terms


def get_representative_samples(
    df: pd.DataFrame,
    tfidf_matrix,
    kmeans: KMeans,
    valid_mask: np.ndarray,
    cluster_id: int,
    n_samples: int = SAMPLES_FOR_NAMING
) -> list:
    """Получение репрезентативных примеров для кластера (ближайших к центроиду)."""
    # Индексы записей в этом кластере
    cluster_mask = df['cluster_id'] == cluster_id
    cluster_indices = df[cluster_mask].index.tolist()

    if len(cluster_indices) == 0:
        return []

    # Индексы в TF-IDF матрице (только для valid записей)
    valid_indices = np.where(valid_mask)[0]
    tfidf_indices = [np.where(valid_indices == idx)[0][0]
                     for idx in cluster_indices if idx in valid_indices]

    if len(tfidf_indices) == 0:
        return []

    # Центроид кластера
    centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)

    # Расстояния до центроида
    cluster_vectors = tfidf_matrix[tfidf_indices]
    distances = cosine_distances(cluster_vectors, centroid).flatten()

    # Сортируем по расстоянию и берем топ-N
    sorted_indices = np.argsort(distances)[:n_samples]

    # Получаем тексты
    selected_df_indices = [cluster_indices[tfidf_indices.index(tfidf_indices[i])]
                           for i in sorted_indices if i < len(tfidf_indices)]

    # Берем оригинальные описания (не очищенные) для LLM
    samples = []
    for idx in cluster_indices:
        if len(samples) >= n_samples:
            break
        text = df.loc[idx, 'request_description']
        if pd.notna(text) and str(text).strip():
            # Обрезаем слишком длинные тексты
            text = str(text)[:500]
            samples.append(text)

    return samples


def name_cluster_with_llm(llm, samples: list, top_terms: list) -> str:
    """Генерация названия кластера через YandexGPT."""
    if not samples:
        return f"Кластер (без примеров)"

    samples_text = "\n---\n".join(samples[:7])  # Берем максимум 7 примеров
    terms_text = ", ".join(top_terms[:10])

    prompt = f"""Ты аналитик техподдержки. Проанализируй примеры обращений пользователей и определи общую тему/домен.

Ключевые слова кластера: {terms_text}

Примеры обращений:
{samples_text}

Напиши ТОЛЬКО краткое название домена/темы (2-5 слов), которое объединяет эти обращения.
Название должно быть на русском языке, без пояснений.

Название домена:"""

    try:
        response = llm.complete(prompt)
        name = response.text.strip()
        # Убираем лишние символы
        name = re.sub(r'^["\'\-\*]+|["\'\-\*]+$', '', name)
        name = name.strip()
        return name if name else f"Кластер (ошибка генерации)"
    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return f"Кластер (ошибка: {str(e)[:30]})"


def name_all_clusters(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    kmeans: KMeans,
    tfidf_matrix,
    valid_mask: np.ndarray
) -> dict:
    """Именование всех кластеров через YandexGPT."""
    print("Инициализация YandexGPT...")

    config = SingleConfig()
    llm = create_llm(model_type='yandexgpt-lite')
    llm.temperature = 0.3

    # Получаем топ-термы
    top_terms = get_top_terms_per_cluster(vectorizer, kmeans)

    cluster_names = {}

    for cluster_id in range(kmeans.n_clusters):
        print(f"Именование кластера {cluster_id + 1}/{kmeans.n_clusters}...")

        samples = get_representative_samples(
            df, tfidf_matrix, kmeans, valid_mask, cluster_id
        )

        name = name_cluster_with_llm(llm, samples, top_terms[cluster_id])
        cluster_names[cluster_id] = name
        print(f"  Кластер {cluster_id}: {name}")

    return cluster_names, top_terms


def save_results(
    df: pd.DataFrame,
    cluster_names: dict,
    top_terms: dict,
    output_file: Path,
    stats_file: Path
):
    """Сохранение результатов в Excel."""
    print(f"Сохранение результатов...")

    # Добавляем названия кластеров
    df['cluster_name'] = df['cluster_id'].map(cluster_names)
    df.loc[df['cluster_id'] == -1, 'cluster_name'] = "Без кластера (пустой текст)"

    # Сохраняем основной результат
    df.to_excel(output_file, index=False)
    print(f"Результаты сохранены в {output_file}")

    # Статистика по кластерам
    stats_data = []
    for cluster_id in sorted(cluster_names.keys()):
        count = (df['cluster_id'] == cluster_id).sum()
        stats_data.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_names[cluster_id],
            'count': count,
            'percentage': round(count / len(df) * 100, 2),
            'top_terms': ', '.join(top_terms[cluster_id])
        })

    # Добавляем записи без кластера
    no_cluster_count = (df['cluster_id'] == -1).sum()
    if no_cluster_count > 0:
        stats_data.append({
            'cluster_id': -1,
            'cluster_name': "Без кластера (пустой текст)",
            'count': no_cluster_count,
            'percentage': round(no_cluster_count / len(df) * 100, 2),
            'top_terms': ''
        })

    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('count', ascending=False)
    stats_df.to_excel(stats_file, index=False)
    print(f"Статистика сохранена в {stats_file}")


def main():
    """Основная функция."""
    print("=" * 60)
    print("Анализ и кластеризация запросов техподдержки")
    print("=" * 60)

    # 1. Загрузка данных
    df = load_data(INPUT_FILE)

    # 2. Предобработка
    df = preprocess_data(df)

    # 3. Векторизация и кластеризация
    df, vectorizer, kmeans, tfidf_matrix, valid_mask = vectorize_and_cluster(df, N_CLUSTERS)

    # 4. Именование кластеров
    cluster_names, top_terms = name_all_clusters(
        df, vectorizer, kmeans, tfidf_matrix, valid_mask
    )

    # 5. Сохранение
    save_results(df, cluster_names, top_terms, OUTPUT_FILE, STATS_FILE)

    print("=" * 60)
    print("Анализ завершен!")
    print(f"Результаты: {OUTPUT_FILE}")
    print(f"Статистика: {STATS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

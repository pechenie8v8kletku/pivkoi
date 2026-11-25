# Business Match

Отдельный Python-сервис B2B-рекомендаций: **компания + до трёх подходящих товаров**,
поиск и онлайн-адаптация без исторического датасета и обучения нейросети.
Python 3.11+, FastAPI, SQLAlchemy, PostgreSQL, Qwen по API, Pydantic AI.

## Запуск

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
# Терминал 1: mock API моделей
.venv\Scripts\python -m business_match.mock_api
# Терминал 2: сервис рекомендаций
.venv\Scripts\python -m uvicorn business_match.app:app --host 0.0.0.0 --port 8000 --workers 1
```

По умолчанию SQLite для локального запуска. Для PostgreSQL поменять `DATABASE_URL`:
`postgresql+psycopg://user:password@localhost:5432/business_match`.
Таблицы создаются при запуске. Mock не скачивает большие модели: embeddings —
детерминированный hashing, rerank — cosine mock, LLM — структурная заглушка.
Он проверяет интеграцию, но не качество рекомендаций. Для настоящих моделей
переключить `MODEL_API_MODE=external` и заполнить реальные API URLs/keys.

Swagger: `/docs`, OpenAPI: `/openapi.json`, health: `/health`.

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```

## Что реализовано

- Полный snapshot клиента и его каталога через `PUT /v1/clients/{id}`.
- Структурированный бизнес-профиль через Pydantic AI с mock или Qwen3.8-27B API.
- Отдельные эмбеддинги потребностей, предложений, товаров; Qwen3-Embedding-4B.
- Каналы поставщиков, покупателей, партнёров, коллег/конкурентов.
- Лента `POST /v1/feed`, семантический поиск `POST /v1/search`.
- BGE reranker после retrieval для feed/search, с fallback на векторный скоринг.
- Мягкий региональный буст; явные регион/город в запросе — жёсткие фильтры.
- Переключаемый `semantic`/`linucb` ranker: semantic cold-start prior и ridge-linear
  персональная модель, обновляемая по положительным действиям.
- Диверсификация по векторам и сегментам; небольшой exploration в ленте.
- Разреженный граф направленных связей сегментов, обновляемый по событиям.
- Проверка привязки событий к выданным карточкам, идемпотентность `event_id`.
- SQL-персистентность профилей, кэша эмбеддингов, выдач, событий.

Синтетического каталога и генератора данных нет. Тестовые фикстуры изолированы в `tests/`.

## Документация

- [Контракт интеграции, события и API провайдеров](docs/integration.md)
- [Алгоритм и границы MVP](docs/ml.md)
- [Каскад, ранкер и mock API моделей](docs/cascade-and-mock.md)

Начальный вариант использует точный перебор векторов, хранимых в SQL JSON.
Это корректный CPU baseline для небольшого каталога, **не ANN/pgvector**.
Переход на pgvector/HNSW — после замера реального объёма и latency.
Запускать один worker: лимитеры API и сериализация обновлений внутрипроцессные.

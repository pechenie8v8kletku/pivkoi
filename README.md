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
- Структурированный бизнес-профиль и поисковые потребности из карточки через Pydantic AI с mock или Qwen3.8-38B API вашего контура (`LLM_MODEL` — точный alias шлюза).
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

Воспроизводимые синтетические сценарии находятся в `scripts/`, регрессионные тесты —
в `tests/`. Сгенерированные каталоги, SQLite и отчёты сохраняются в игнорируемой Git
папке `local_experiments/`.

## Документация

- [Контракт интеграции, события и API провайдеров](docs/integration.md)
- [Алгоритм и границы MVP](docs/ml.md)
- [Каскад, ранкер и mock API моделей](docs/cascade-and-mock.md)
- [Синтетический эксперимент и найденные ошибки](docs/synthetic-validation.md)
- [Смешанный рынок, генерация потребностей и смена интереса](docs/mixed-market-validation.md)
- [Масштабирование, настройки и измерения](docs/scaling.md)

Текущий backend читает SQL JSON страницами, считает близость блоками NumPy и
передаёт в ранкер только top-K компаний каждой роли через `CandidateRetriever`.
Это корректный CPU baseline для небольшого каталога, **не ANN/pgvector**.
Переход на pgvector/HNSW — после замера реального объёма и latency.
Запускать один worker: лимитеры API и сериализация обновлений внутрипроцессные.

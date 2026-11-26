# Интеграция

## Что понадобится на стенде

1. PostgreSQL: DSN, пользователь с правами создать четыре таблицы `bm_*`.
2. URL OpenAI-совместимого `/v1` и ключ для Qwen3.8-38B и Qwen3-Embedding-4B.
3. Для LLM — поддержка tool calling для структурированного результата Pydantic AI.
   Если шлюз допускает только JSON text/JSON schema, адаптировать `Providers.profile`.
4. Опциональный URL BGE reranker и точный контракт шлюза.
5. Доступ из сервиса к этим API; `SERVICE_API_KEY` для межсервисных запросов.
6. Устойчивые строковые идентификаторы клиентов и товаров, snapshot каталога.

`Qwen3.8-38B` — обозначение модели в вашем контуре; задайте в `LLM_MODEL` точный alias шлюза.
LLM через Pydantic AI генерирует из карточки `needs`: конкретные товары/услуги для поиска
и направления связи. `products` и `explicit_needs` необязательны. Предположения имеют
`evidence=inferred` и пониженный вес, а не становятся реальным ассортиментом/SKU клиента.
Профиль строится при onboarding и смысловых обновлениях; клики обновляют ранкер без LLM.
После смены модели или версии профиля повторно отправьте snapshots клиентов.
Эмбеддер мультиязычный; русский поддерживается. Размерность берём из ответа,
не фиксируем вручную. При смене embedding-модели переиндексировать **всех** клиентов.

## Режимы моделей и ranker

- `MODEL_API_MODE=mock` направляет адаптеры на `MOCK_API_BASE_URL`:
  OpenAI-compatible `/v1/embeddings`, `/v1/chat/completions` и `/rerank`.
- `MODEL_API_MODE=external` использует отдельные LLM, embeddings и rerank URL/keys.
- `RANKER_MODE=linucb` включает персональное онлайн-обновление; `semantic` оставляет
  чистый cold-start ranking. Переключатели модели не меняют контракт API.
- `RERANK_ENABLED=true` включает BGE adapter; `RERANK_FEED_ENABLED=true` разрешает
  cross-encoder в ленте. Количество пар ограничивает `RERANK_TOP_K`.
- Настройки свежести, bandit, diversity и exploration описаны в `.env.example`.

Запустить mock:
```powershell
.venv\Scripts\python -m business_match.mock_api
```
Mock endpoints: `GET /health`, `POST /v1/embeddings`, `POST /v1/chat/completions`,
`POST /rerank`. Embeddings используют детерминированный hashing vector, LLM —
структурную заглушку из полей клиента, reranker — cosine hash similarity.
Это не имитация качества Qwen/BGE и не подходит для offline evaluation.

Rerank mock contract — пара query/offer для feed и search:
```json
{"model":"bge-reranker-v2-m3","pairs":[
  {"query":"пекарне нужен поставщик муки","document":"мука оптом с доставкой"}
]}
```
Ответ: `{"results":[{"index":0,"relevance_score":0.8}]}`. Для отличающегося
реального API адаптировать только `Providers.rerank_pairs`.

## Входные данные

`PUT /v1/clients/{id}` — полный текущий снимок. Отсутствующие товары удаляются из
индекса клиента; `active=false` выключает клиента. Для удаления товара можно передать
`active=false` или убрать из snapshot. `stock=null` означает «неизвестно», `stock=0`
выключает товар из подбора. Профиль компании остаётся доступным даже без товаров.

```json
{
  "id": "client-123",
  "name": "Название компании",
  "description": "Чем занимается бизнес",
  "segment": "Пекарня",
  "region": "Татарстан",
  "city": "Казань",
  "explicit_needs": ["Мука оптом с доставкой"],
  "products": [
    {"id": "sku-1", "name": "Хлеб", "description": "Пшеничный",
     "category": "Выпечка", "stock": 20, "price": 90, "unit": "шт", "active": true}
  ],
  "active": true
}
```

Клиент должен иметь хотя бы название; неизвестный сегмент — `unknown`.
Поля витрины маппятся в `ClientInput`, специфические поля 1С адаптируются снаружи.
Один запрос до 5000 товаров; это ограничение MVP, большие каталоги требуют chunked ingestion.
Endpoint синхронный: возвращается после профиля/индексации, может ждать лимитер.
Вызывайте из фонового процесса интеграции с достаточным HTTP timeout.
Повторный идентичный snapshot не вызывает модели. Изменения остатков/цен/географии
не запускают LLM. Семантические изменения — запускают; эмбеддинги кэшируются по тексту/модели.
Загрузка snapshot атомарно заменяет профиль только после успешного embedding.

## Выдача

`POST /v1/feed`:
```json
{"client_id":"client-123","limit":20,"exclude_client_ids":[]}
```

`POST /v1/search`:
```json
{"client_id":"client-123","query":"поставщик муки","limit":20}
```

Опционально: `intent` = `supplier|buyer|partner|peer`, `region`, `city`, `seed`.
Поиск по умолчанию рассматривает все каналы; UI может явно выбрать нужную роль.
`region/city` в запросе — точное совпадение строк, нормализовать на стороне интеграции.
В ответе `request_id`, `items`, `mode`, `policy_version`, `diagnostics`.
Карточка: `client_id`, `name`, `segment`, география, `intent`, `score`, `reason`,
`matched_need`, `matched_text`, `products`, `exploration`.
`score` — технический скоринг, не вероятность сделки и не калиброванная релевантность.
Пагинация MVP — новый запрос с `exclude_client_ids` уже загруженных карточек.
Повтор без exclusions может вернуть те же компании; рекомендация не считается показом.

## Фронтенд → события

`POST /v1/events`, заголовок `X-API-Key` при настроенном ключе.
Сохранять `request_id` конкретной выдачи и генерировать UUID события **один раз**,
повторять доставку с прежним UUID.

```json
{"events":[{
  "event_id":"stable-uuid",
  "request_id":"uuid-from-feed",
  "client_id":"client-123",
  "target_client_id":"client-456",
  "kind":"impression",
  "visible_fraction":0.8,
  "dwell_ms":1000
}]}
```

- `impression`: >=50% карточки видимо >=1 секунды (время проверяет фронт), не по факту загрузки.
- `dwell`: суммарное активное время видимости; не учитывать background вкладку.
- `open`: открыл карточку компании/товара.
- `like`, `unlike`: явные действия. В MVP unlike нейтрален, не отменяет ранее учтённый like.
- `contact`: нажал связаться/открыл контакт, не считать подтверждённой сделкой.
- `hide`: явное скрытие, исключает компанию из выдачи на 7 дней.
- `product_id` опционален; должен быть среди товаров выданной карточки.
- `occurred_at` опционален, ISO8601 с часовым поясом; без него используется серверное время.

До 100 событий за раз. Результат по каждому: `accepted|duplicate|rejected` с `detail`.
HTTP 200 не означает принятие всех событий — проверять `results`.
Отсутствие клика/быстрый скролл не трактуются как отрицательная награда.
Повторные реакции ограничены одной единицей свидетельства на компанию в сутки.
Нет обязательного предварительного impression для open/contact: некоторые UI открывают
карточку до отправки пакетного impression. Событие всё равно обязано ссылаться на выдачу.

## Провайдеры

Эмбеддинги: `POST {EMBEDDING_BASE_URL}/embeddings`, стандартные `model`, `input`,
ответ `data: [{index, embedding}]`. Query instruction добавляется в текст в формате Qwen3.
Offers/products без инструкции; формат уточнить у шлюза во избежание двойного prefix.

В `external` режиме reranker не является стандартной OpenAI-ручкой; используется
вышеописанный batch pair контракт. При ошибке или отключённом rerank идёт semantic fallback.

Лимиты по умолчанию: embedding/rerank 28 RPM, 60000 estimated TPM, LLM 9 RPM.
Оценка токенов консервативная по символам, не точный tokenizer; оставлен запас.
HTTP retry автоматически не выполняется: интеграция повторяет snapshot при 502/ошибке.
Другие приложения с тем же ключом делят реальные лимиты шлюза: при необходимости
понизить конфигурацию или вынести лимитер в общий Redis.

## Сборка сервисов и эксплуатация

- Backend должен проверять права пользователя на `client_id`; ключ сервиса не заменяет auth пользователя.
- Airflow/DAG/CDC пока не встроены. Их задача — периодически посылать актуальный snapshot.
- Схема создаётся `create_all`; при изменении схемы между версиями нужны миграции.
- Граф глобальных связей сегментов хранится в `bm_relation_edges` и
  `bm_relation_observations`; это MVP aggregate, перед production вводить Alembic migration.
- Embedding-cache и история не очищаются автоматически. Настроить retention/maintenance
  после выбора SLA повторной доставки; выдачи должны жить не меньше окна доставки событий.
- API не возвращает embeddings. История и каталог переживают перезапуск процесса.
- Статический OpenAPI можно получить из работающего `/openapi.json`.

# Развёртывание в Kubernetes (on-prem)

Манифесты в этой папке проверены только сборкой `kubectl kustomize k8s/` (11 объектов, без ошибок).
На кластере не разворачивались: кластера в среде разработки нет. Тот же набор сервисов проверен
в Docker Compose (`docker-compose.yml` в корне проекта).

## Состав

| Объект | Что это | Зачем |
|---|---|---|
| `Namespace goszakup` | пространство имён | изоляция проекта |
| `ConfigMap goszakup-config` | `LLM_BASE_URL`, `LLM_MODEL`, `TZ`, параметры обновления | настройки без пересборки образа |
| `Secret goszakup-secrets` | `GOSZAKUP_TOKEN`, `POSTGRES_PASSWORD`, `DATABASE_URL` | создаётся вручную (см. `secret.example.yaml`); в образ и git токен не попадает |
| `StatefulSet postgres` + headless `Service` | PostgreSQL 16, PVC 20 Gi | хранилище: таблицы, витрины, журнал `etl_runs`, водяные знаки |
| `Deployment ollama` + `Service` + PVC 20 Gi | локальная LLM (qwen2.5:7b), `OLLAMA_KEEP_ALIVE=-1` | маршрутизация вопросов; данные не уходят из контура |
| `Deployment agent` (2 реплики) + `Service` | FastAPI: `POST /ask`, `GET /health` | агент без состояния, масштабируется горизонтально |
| `CronJob updater` + PVC `raw-data` 10 Gi | `python update.py` каждые 6 ч (`Asia/Almaty`) | инкрементальное обновление из API OWS v3, задержка ≤ 24 ч |

Агент и планировщик обновлений собраны из одного образа: `Dockerfile` в корне, python:3.12-slim,
пользователь 10001, около 288 МБ.

## Почему так

- **CronJob вместо постоянного процесса-планировщика.** `concurrencyPolicy: Forbid` и
  `pg_advisory_lock` внутри `update.py` исключают параллельные запуски. `activeDeadlineSeconds: 21600`
  рассчитан на первую полную выгрузку, которая заняла 3,6 ч. Обычный инкремент идёт около 25 минут.
  Сбой задания не теряет данные: транзакция ETL откатывается, водяные знаки не двигаются,
  и следующий запуск забирает то же окно.
- **PVC `raw-data`.** ETL пересобирает базу из сырых снимков API, поэтому это постоянное
  хранилище, а не кэш. Доступ ReadWriteOnce: им пользуется только `updater`.
- **Пробы агента на `/health`.** Ответ 200 приходит и при устаревших данных: устаревание видно
  в поле `data_stale` и разбирается мониторингом, а не перезапуском пода. `failureThreshold: 5` взят
  потому, что во время ETL запросы к витринам ждут около 25–30 с.
- **LLM.** На CPU qwen2.5:7b отвечает за 4–8 с на вопрос (замер в Docker на Mac, 10 ядер); первый
  запрос после загрузки модели около 130 с. Для эксплуатации нужен узел с GPU: раскомментировать
  `nvidia.com/gpu: 1` в `ollama.yaml`. Модель меняется через `LLM_BASE_URL` и `LLM_MODEL` без пересборки
  образа: подходит любой OpenAI-совместимый сервер, например vLLM.
- **Без доступа в интернет.** Init-контейнер `pull-model` скачивает модель только при первом запуске.
  В закрытом контуре модель заранее кладётся в PVC `ollama-models`, образы — во внутренний реестр
  (поле `images` в `kustomization.yaml`).

## Запуск

```bash
kubectl apply -f k8s/namespace.yaml
kubectl -n goszakup create secret generic goszakup-secrets \
  --from-literal=GOSZAKUP_TOKEN=<токен> \
  --from-literal=POSTGRES_PASSWORD=<пароль> \
  --from-literal=DATABASE_URL=postgresql://postgres:<пароль>@postgres:5432/goszakup
kubectl apply -k k8s/
```

**Первичное наполнение.** Первый запуск CronJob на пустой базе выполняет полную выгрузку (около 3,6 ч).
Путь «пустая база → `update.py`» проверен в контейнере на одном БИН. Чтобы не ждать, можно запустить
задание сразу:

```bash
kubectl -n goszakup create job --from=cronjob/updater updater-init
```

Либо восстановить базу из дампа (`pg_restore` в под postgres) и скопировать `data/raw` в PVC `raw-data`;
тогда первый запуск будет инкрементальным.

## Не сделано

- Нет Ingress или TLS, авторизации на `/ask`, NetworkPolicy, резервного копирования PostgreSQL
  по расписанию, мониторинга (`/health.data_stale`, `etl_runs.status`).
- Манифесты не проверены на реальном кластере.

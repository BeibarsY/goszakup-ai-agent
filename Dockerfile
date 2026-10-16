# Один образ для агента (FastAPI) и обновления данных (update.py / scheduler.py)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # update.py пишет отметки времени по местному времени, как indexDate в API OWS
    TZ=Asia/Almaty

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py *.sql bins.txt eval_cases.json cpi_template.csv cpi.csv ./

# без root; data/ — сырые снимки API (том), logs/ — журналы
RUN useradd --system --uid 10001 --home /app app \
    && mkdir -p data logs && chown -R app:app /app
USER app

EXPOSE 8080
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8080"]

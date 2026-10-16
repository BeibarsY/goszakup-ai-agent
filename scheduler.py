"""
Планировщик обновления для контейнера updater (docker compose): запускает update.py сразу при старте,
затем каждые UPDATE_INTERVAL_HOURS часов (по умолчанию 6) от начала предыдущего запуска.
Сбой одного запуска не останавливает планировщик: update.py сам пишет ошибку в etl_runs и не двигает
водяные знаки, следующий запуск заберёт то же окно. Два запуска одновременно не пойдут
(pg_advisory_lock внутри update.py), поэтому ручной `python update.py` рядом безопасен.

В Kubernetes вместо планировщика — CronJob, который запускает update.py напрямую (см. k8s/).
"""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

INTERVAL = timedelta(hours=float(os.getenv("UPDATE_INTERVAL_HOURS", "6")))
RUN_ON_START = os.getenv("UPDATE_ON_START", "1") == "1"
stopping = False


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [scheduler] {msg}", flush=True)


def stop(signum, _frame):
    global stopping
    stopping = True
    log(f"получен сигнал {signum}, завершаюсь после текущего шага")


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    next_run = datetime.now() if RUN_ON_START else datetime.now() + INTERVAL
    log(f"интервал {INTERVAL}, первый запуск {next_run:%Y-%m-%d %H:%M}")
    while not stopping:
        if datetime.now() >= next_run:
            started = datetime.now()
            rc = subprocess.run([sys.executable, "update.py"]).returncode
            log(f"update.py завершился с кодом {rc} за {datetime.now() - started}")
            next_run = started + INTERVAL
            if next_run <= datetime.now():   # запуск дольше интервала: следующий сразу не начинаем
                next_run = datetime.now() + timedelta(minutes=5)
            log(f"следующий запуск {next_run:%Y-%m-%d %H:%M}")
        time.sleep(10)


if __name__ == "__main__":
    main()

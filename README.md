# hft-lab

Учебная лаборатория для микроструктурной стратегии (mean reversion) на крипто-фьючерсах OKX.
Сейчас готов этап 0: запись данных, локальный стакан и проверка записи. Торговли здесь нет.

Для работы с Claude Code: откройте эту папку и запустите `claude`. Он прочитает `CLAUDE.md` и продолжит с нужного места.
Установка Claude Code: [официальная инструкция](https://code.claude.com/docs/en/quickstart).

## Установка
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                 # должно быть 22 passed
```

## Первый запуск рекордера (делается руками, один раз)
```
python -m hftlab.recorder --inst BTC-USDT-SWAP --out data
```
Подождать 10–15 минут, остановить через Ctrl+C и проверить:
```
python -m hftlab.check_recording data
```
Что должно быть в отчёте:
- `books` и `trades` с ненулевым числом сообщений, `snapshots >= 1`;
- `gap_events == gaps_detected` (если не равны, логика расходится с реальностью, разбираем с Claude Code);
- `crossed books: 0`;
- задержка `recv-ts` порядка десятков миллисекунд; отрицательная значит расхождение часов (проверьте синхронизацию времени).

Если подключение не открывается, возможно, OKX недоступен с вашего IP или из вашей сети: это надо выяснить сразу,
а запасной вариант (Gate) потребует переписать подключение и разбор стакана.

## Структура
```
CLAUDE.md                    контекст и правила для Claude Code
docs/ROADMAP.md             этапы с критериями выхода
docs/GLOSSARY.md            термины из ваших списков по приоритету
docs/source/                исходный список терминов, как был
src/hftlab/okx_book.py      стакан, microprice, imbalance
src/hftlab/storage.py       запись и чтение gzip по часам
src/hftlab/recorder.py      WebSocket-рекордер
src/hftlab/check_recording.py  отчёт по записи
src/hftlab/runs.py          журнал бэктестов (SQLite)
src/hftlab/vault_export.py  экспорт журнала в Obsidian
results/                    runs.sqlite (в git не попадает)
tests/                      тесты на синтетических данных
data/                       записанные данные (в git не попадают)
```

## Журнал бэктестов и Obsidian
Каждый прогон бэктеста записывается в `results/runs.sqlite` (пример в начале `src/hftlab/runs.py`).
Потом выгрузка в ваше хранилище Obsidian:
```
python -m hftlab.vault_export --vault "C:/Users/<вы>/Obsidian/<хранилище>/hft-lab"
```
Откройте `hft-lab/Home.md`. Панели в `Dashboards/` работают в Obsidian 1.9+ без плагинов.
Свои мысли пишите ниже строки `AUTO:END`: при следующей выгрузке они сохранятся.
`results/runs.sqlite` в git не попадает: копируйте его в бэкап вместе с хранилищем.

## Про деньги
Ключи API сюда не вставляются и в репозиторий не попадают (`.env` и `*.key` в `.gitignore`).
Реальные ордера только после этапов 4–5 из `docs/ROADMAP.md`.

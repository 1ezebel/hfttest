# hft-lab

Учебная лаборатория для микроструктурной стратегии (mean reversion) на крипто-фьючерсах OKX.
Готов код этапов 0-5: запись данных, признаки, проверка сигналов, симулятор исполнения, учёт ордеров
с лимитами риска и бумажная торговля. Реальной торговли здесь нет, ключи API не нужны.

Для работы с Claude Code: откройте эту папку и запустите `claude`. Он прочитает `CLAUDE.md` и продолжит с нужного места.
Установка Claude Code: [официальная инструкция](https://code.claude.com/docs/en/quickstart).

## Установка
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # ставит зависимости и сам проект (папку src)
python -m pytest                 # должно быть 65 passed
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

Если порт 8443 закрыт вашей сетью, добавьте `--url wss://ws.okx.com:443/ws/v5/public`.
Если подключение не открывается совсем, возможно, OKX недоступен с вашего IP или из вашей сети: это надо выяснить сразу,
а запасной вариант (Gate) потребует переписать подключение и разбор стакана.

## Полный цикл (когда есть запись)
```
python -m hftlab.features data --out processed           # 1. сырьё -> таблицы с признаками
python -m hftlab.research processed --strategy mr_ema    # 2. есть ли у сигнала предсказательная сила
python -m hftlab.sim processed --strategy mr_ema         # 3. выживает ли он при пассивном исполнении
python -m hftlab.paper --strategy mr_ema --params '{"halflife_s": 5, "z_entry": 2}' --minutes 60   # 4. вживую, без денег
python -m hftlab.paper_compare results/paper/<файл>.jsonl   # 5. совпал ли paper с симулятором
```
Шаги 2-4 сами пишут результаты в журнал `results/runs.sqlite`.
Долгая запись на VPS или ноутбуке: `deploy/README.md`.

## Структура
```
CLAUDE.md                    контекст и правила для Claude Code
docs/ROADMAP.md              этапы с критериями выхода
docs/GLOSSARY.md             термины по приоритету
deploy/                      автозапуск рекордера (systemd, .bat)
src/hftlab/okx_book.py       стакан, microprice, imbalance
src/hftlab/storage.py        запись и чтение gzip по часам
src/hftlab/recorder.py       WebSocket-рекордер с аудитом стакана
src/hftlab/check_recording.py  отчёт по записи
src/hftlab/health.py         жива ли запись
src/hftlab/features.py       признаки -> parquet
src/hftlab/research.py       сигналы, markout, walk-forward
src/hftlab/decay.py          затухание сигнала (half-life) с графиком
src/hftlab/sim.py            симулятор пассивного исполнения
src/hftlab/universe.py       у каких фьючерсов спред шире комиссий
src/hftlab/oms.py            ордера, лимиты риска, kill switch
src/hftlab/paper.py          бумажная торговля
src/hftlab/paper_compare.py  paper против симулятора
src/hftlab/runs.py           журнал бэктестов (SQLite)
src/hftlab/vault_export.py   экспорт журнала в Obsidian
tests/                       65 тестов
data/                        сырая запись (в git не попадает)
processed/                   таблицы с признаками (в git не попадают, пересчитываются)
results/                     журнал и логи paper (в git не попадают, бэкапить)
```

## Журнал бэктестов и Obsidian
Каждый прогон бэктеста записывается в `results/runs.sqlite` (пример в начале `src/hftlab/runs.py`).
Потом выгрузка в ваше хранилище Obsidian:
```
python -m hftlab.vault_export --vault "C:/Users/<вы>/Obsidian/<хранилище>/hft-lab"
```
В архиве уже лежит готовая выгрузка первой сессии: скопируйте папку `obsidian/hft-lab` в своё хранилище
(в ней есть запись дневника за 10.11.2026). Откройте `hft-lab/Home.md`. Панели в `Dashboards/` работают в Obsidian 1.9+ без плагинов.
Свои мысли пишите ниже строки `AUTO:END`: при следующей выгрузке они сохранятся.
`results/runs.sqlite` в git не попадает: копируйте его в бэкап вместе с хранилищем.

## Про деньги
Ключи API сюда не вставляются и в репозиторий не попадают (`.env` и `*.key` в `.gitignore`).
Реальные ордера только после этапов 4–5 из `docs/ROADMAP.md`.

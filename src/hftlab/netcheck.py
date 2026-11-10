"""Диагностика связи с OKX: почему данные приходят с опозданием.

    python -m hftlab.netcheck            # занимает около 40 секунд

Запускать, когда check_recording показывает большую задержку (latency больше секунды).
Рекордер при этом можно не останавливать.

Что проверяется:
  1. Часы компьютера: сравниваем с временем биржи через обычный запрос.
     Если часы врут на секунду, вся "задержка" в отчётах сдвинута на эту секунду.
  2. Канал: 30 секунд читаем тот же поток, что рекордер, но НИЧЕГО с ним не делаем
     (не разбираем стакан, не пишем на диск). Если опоздание растёт даже так,
     виноват канал (провайдер, VPN, Wi-Fi), а не наш код.
  3. Процессор: какую долю времени заняла сама программа.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
import urllib.request

from hftlab.recorder import URL, sub_msg

REST_TIME = "https://www.okx.com/api/v5/public/time"
TS_RE = re.compile(r'"ts":"(\d{13})"')  # время биржи достаём без полного разбора JSON: так быстрее


def clock_offset(n: int = 5) -> tuple[float, float]:
    """(на сколько мс наши часы отстают от биржи, время запроса туда-обратно в мс).

    Плюс значит, что часы компьютера отстают; минус, что спешат.
    Берём попытку с самым быстрым ответом: у неё оценка точнее всего.
    """
    best = None
    for _ in range(n):
        t0 = time.time()
        req = urllib.request.Request(REST_TIME, headers={"User-Agent": "hftlab"})
        with urllib.request.urlopen(req, timeout=15) as r:
            server_ms = float(json.load(r)["data"][0]["ts"])
        t1 = time.time()
        rtt = (t1 - t0) * 1000
        offset = server_ms - (t0 + t1) / 2 * 1000
        if best is None or rtt < best[1]:
            best = (offset, rtt)
    return best


async def stream_lag(url: str, insts: list[str], seconds: float, offset_ms: float) -> dict:
    import websockets

    lags: list[tuple[float, float]] = []  # (секунда от старта, опоздание в мс)
    n_msgs = n_bytes = 0
    async with websockets.connect(url, ping_interval=None, max_size=2**23, open_timeout=15) as ws:
        for i in insts:
            await ws.send(sub_msg("subscribe", "books", i))
            await ws.send(sub_msg("subscribe", "trades", i))
        start = time.monotonic()
        cpu0 = time.process_time()
        while time.monotonic() - start < seconds:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            now_ms = time.time() * 1000
            n_msgs += 1
            n_bytes += len(raw)
            m = TS_RE.search(raw[-400:]) or TS_RE.search(raw)
            if m and len(raw) < 100_000:  # первый snapshot огромный и не показателен
                lags.append((time.monotonic() - start, now_ms + offset_ms - float(m.group(1))))
        elapsed = time.monotonic() - start
        cpu = time.process_time() - cpu0
    first = [l for s, l in lags if s < elapsed / 3]
    last = [l for s, l in lags if s > elapsed * 2 / 3]
    # Опоздание по десяти равным отрезкам времени: видно, растёт оно или стоит на месте
    step = elapsed / 10
    timeline = []
    for k in range(10):
        part = [l for s, l in lags if k * step <= s < (k + 1) * step]
        timeline.append(statistics.median(part) if part else None)
    return {
        "timeline_ms": timeline, "step_s": step,
        "seconds": elapsed, "msgs_per_s": n_msgs / elapsed, "kb_per_s": n_bytes / elapsed / 1024,
        "cpu_share": cpu / elapsed,
        "lag_first_ms": statistics.median(first) if first else None,
        "lag_last_ms": statistics.median(last) if last else None,
        "lag_min_ms": min(l for _, l in lags) if lags else None,
    }


def verdict(offset_ms: float, rtt_ms: float, r: dict) -> list[str]:
    out = []
    if abs(offset_ms) > 250:
        way = "отстают" if offset_ms > 0 else "спешат"
        out.append(f"ЧАСЫ: компьютер {way} на {abs(offset_ms) / 1000:.1f} с. Включите автоматическую "
                   "синхронизацию времени (Параметры -> Время и язык -> Дата и время -> Синхронизировать).")
    first, last = r["lag_first_ms"], r["lag_last_ms"]
    if first is None or last is None:
        return out + ["ПОТОК: сообщений не пришло, проверить нечем."]
    # рост опоздания в секундах за секунду записи: 0.01 = за час набежит 36 секунд
    slope = (last - first) / 1000 / (r["seconds"] * 2 / 3)
    if slope > 0.005 and last - first > 300:
        if r["cpu_share"] > 0.7:
            out.append("ПРОЦЕССОР: программа занята почти всё время и не успевает за потоком. "
                       "Закройте тяжёлые программы или запишите меньше инструментов.")
        else:
            out.append(f"КАНАЛ: опоздание выросло с {first / 1000:.1f} до {last / 1000:.1f} с за "
                       f"{r['seconds']:.0f} с (за час набежит около {slope * 3600:.0f} с), хотя программа "
                       "простаивала. Поток не пролезает через сеть: "
                       "отключите VPN/прокси, попробуйте кабель вместо Wi-Fi или другую сеть. "
                       "Если не поможет, записывать нужно с VPS.")
    elif last > 1000:
        out.append(f"КАНАЛ: опоздание постоянное, около {last / 1000:.1f} с (не растёт). "
                   "Запись для исследования годится, для бумажной торговли нет.")
    if rtt_ms > 800:
        out.append(f"СЕТЬ: обычный запрос к бирже идёт {rtt_ms:.0f} мс, это очень медленно.")
    return out or [f"ВСЁ В ПОРЯДКЕ: опоздание около {last:.0f} мс и не растёт."]


def main() -> int:
    ap = argparse.ArgumentParser(description="Диагностика связи с OKX")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--inst", action="append")
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()
    insts = a.inst or ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]

    print("1/2 часы и время запроса...")
    rest_error = None
    try:
        offset, rtt = clock_offset()
        print(f"    часы компьютера относительно биржи: {-offset:+.0f} мс (минус = отстают)")
        print(f"    запрос туда-обратно: {rtt:.0f} мс")
    except Exception as e:  # noqa: BLE001 - сам отказ и есть результат диагностики
        rest_error, offset, rtt = e, 0.0, 0.0
        print(f"    сайт биржи www.okx.com НЕ ОТВЕТИЛ за 15 с: {e}")
        print("    часы проверить не удалось, опоздание ниже посчитано без поправки на часы")
    print(f"2/2 слушаю поток {a.seconds:.0f} с, ничего с ним не делая...")
    r = asyncio.run(stream_lag(a.url, insts, a.seconds, offset))
    print(f"    сообщений в секунду: {r['msgs_per_s']:.0f}, объём: {r['kb_per_s']:.0f} КБ/с, "
          f"загрузка процессора программой: {r['cpu_share'] * 100:.0f}%")
    f = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
    print(f"    опоздание (с поправкой на часы): в начале {f(r['lag_first_ms'])} мс, "
          f"в конце {f(r['lag_last_ms'])} мс, минимум {f(r['lag_min_ms'])} мс")
    print("    опоздание по ходу замера, мс: "
          + "  ".join("-" if v is None else f"{v:.0f}" for v in r["timeline_ms"])
          + f"   (шаг {r['step_s']:.0f} с)")
    print()
    if rest_error is not None:
        print("ДОСТУП: обычный запрос к www.okx.com не проходит. Откройте https://www.okx.com в браузере: "
              "если сайт не открывается или грузится очень долго, ваша сеть плохо пропускает OKX. "
              "Тогда запись с этого компьютера ненадёжна, нужен другой канал или VPS.")
    for line in verdict(offset, rtt, r):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

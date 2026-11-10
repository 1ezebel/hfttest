"""Рекордер публичных данных OKX: стакан (`books`) и сделки (`trades`).

Запуск:
    python -m hftlab.recorder --inst BTC-USDT-SWAP --inst ETH-USDT-SWAP --out data

Ключи API не нужны: каналы публичные.

Что делает:
  * пишет КАЖДОЕ сообщение как есть + наше время получения (recv_ns);
  * параллельно ведёт стакан и проверяет цепочку seqId; при пропуске пишет
    событие #GAP и пересоздаёт подписку на books этого инструмента;
  * держит соединение: "ping" раз в 20 секунд (OKX рвёт молчащие соединения),
    сторожевой таймер на тишину, переподключение с экспоненциальной задержкой.

ВАЖНО: код написан по документации OKX API v5 и покрыт тестами на синтетике,
но живую биржу из среды разработки проверить было нельзя. Первый запуск надо
сделать руками и посмотреть лог (см. README, раздел "Первый запуск").
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from collections import Counter

import websockets

from hftlab.okx_book import OrderBook, Status
from hftlab.storage import HourlyWriter

URL = "wss://ws.okx.com:8443/ws/v5/public"
# Тот же сервер слушает и 443: пригодится, если сеть/прокси режет нестандартный порт 8443
URL_443 = "wss://ws.okx.com:443/ws/v5/public"
PING_EVERY_S = 20
SILENCE_TIMEOUT_S = 30
FLUSH_EVERY_S = 5
STATS_EVERY_S = 60

log = logging.getLogger("recorder")


def sub_msg(op: str, channel: str, inst_id: str) -> str:
    return json.dumps({"op": op, "args": [{"channel": channel, "instId": inst_id}]})


async def _ping_loop(ws) -> None:
    while True:
        await asyncio.sleep(PING_EVERY_S)
        await ws.send("ping")


async def _housekeeping(writer: HourlyWriter, counts: Counter) -> None:
    last_stats = time.monotonic()
    while True:
        await asyncio.sleep(FLUSH_EVERY_S)
        writer.flush()
        if time.monotonic() - last_stats >= STATS_EVERY_S:
            last_stats = time.monotonic()
            log.info("messages so far: %s", dict(counts))


async def run_once(insts: list[str], writer: HourlyWriter, counts: Counter, url: str = URL) -> None:
    async with websockets.connect(
        url, ping_interval=None, max_size=2**23, open_timeout=15
    ) as ws:
        books = {i: OrderBook(i) for i in insts}
        now = time.time_ns()
        for i in insts:
            writer.write_event(i, now, "CONNECT")
        for i in insts:
            await ws.send(sub_msg("subscribe", "books", i))
            await ws.send(sub_msg("subscribe", "trades", i))
        log.info("connected, subscribed to %s", insts)

        tasks = [asyncio.create_task(_ping_loop(ws))]
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=SILENCE_TIMEOUT_S)
                recv_ns = time.time_ns()
                if raw == "pong":
                    continue
                msg = json.loads(raw)

                if "event" in msg:  # ответы на подписку, ошибки
                    if msg["event"] == "error":
                        raise RuntimeError(f"OKX error: {raw}")
                    log.info("event: %s", raw)
                    continue

                arg = msg.get("arg", {})
                inst, channel = arg.get("instId"), arg.get("channel")
                if inst not in books:
                    continue

                writer.write_raw(inst, recv_ns, raw)  # сначала сохраняем сырьё
                counts[f"{inst}/{channel}"] += 1

                if channel == "books":
                    for d in msg["data"]:
                        status = books[inst].apply(msg["action"], d)
                        if status in (Status.GAP, Status.RESET):
                            log.warning("%s: %s, resubscribing books", inst, status.value)
                            writer.write_event(inst, recv_ns, "GAP", status.value)
                            counts[f"{inst}/gaps"] += 1
                            await ws.send(sub_msg("unsubscribe", "books", inst))
                            await ws.send(sub_msg("subscribe", "books", inst))
                            break
        finally:
            for t in tasks:
                t.cancel()


async def main_async(insts: list[str], out: str, url: str = URL) -> None:
    writer = HourlyWriter(out, "okx")
    counts: Counter = Counter()
    house = asyncio.create_task(_housekeeping(writer, counts))
    attempt = 0
    try:
        while True:
            started = time.monotonic()
            try:
                await run_once(insts, writer, counts, url)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - любая причина => переподключаемся
                log.error("connection lost: %r", e)
            if time.monotonic() - started > 60:
                attempt = 0  # соединение жило долго: сбрасываем задержку
            delay = min(60.0, 2.0**attempt) + random.uniform(0, 1)  # exponential backoff + jitter
            attempt += 1
            log.info("reconnect in %.1fs", delay)
            await asyncio.sleep(delay)
    finally:
        house.cancel()
        writer.close()


def main() -> None:
    p = argparse.ArgumentParser(description="OKX public data recorder (books + trades)")
    p.add_argument("--inst", action="append", help="instId, можно несколько раз")
    p.add_argument("--out", default="data", help="папка для записи")
    p.add_argument("--url", default=URL, help=f"адрес WS; если 8443 закрыт: {URL_443}")
    args = p.parse_args()
    insts = args.inst or ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(main_async(insts, args.out, args.url))
    except KeyboardInterrupt:
        log.info("stopped by user")


if __name__ == "__main__":
    main()

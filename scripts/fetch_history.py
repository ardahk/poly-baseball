"""Cache historical MLB state timelines to disk so models can be refit cheaply.

The MLB Stats API gives us ~2,430 finished games per season with a full play-by-play
state timeline. That is ~675x more data than the 16 games of market ticks we have
collected, and it is where model SELECTION must happen -- picking among many models
on 16 games would just select the luckiest one.

Writes artifacts/history/<season>.jsonl, one compact record per game:
  {"pk": 745804, "y": 1, "s": [[inning, is_top, outs, hs, as, b1, b2, b3], ...]}

Run: ./.venv/bin/python scripts/fetch_history.py 2022 2023 2024 2025
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polybot import backtest, mlb

OUT_DIR = "artifacts/history"


def fetch_season(year: int) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{year}.jsonl")
    done: set[int] = set()
    if os.path.exists(path):                       # resume, don't refetch
        with open(path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["pk"])
                except Exception:
                    pass

    client = mlb.MLBClient()
    scheduled = [g for g in client.schedule(f"{year}-03-15", f"{year}-11-15")
                 if g["status"] == "Final"]
    todo = [g for g in scheduled if g["game_pk"] not in done]
    print(f"{year}: {len(scheduled)} final games, {len(done)} cached, "
          f"{len(todo)} to fetch", flush=True)

    t0 = time.time()
    written = 0
    with open(path, "a") as fh:
        for i, g in enumerate(todo, 1):
            timeline, home_won = backtest.build_state_timeline(client, g["game_pk"])
            if timeline and home_won is not None:
                states = [[gs.inning, int(gs.is_top), gs.outs, gs.home_score,
                           gs.away_score, int(gs.on_first), int(gs.on_second),
                           int(gs.on_third)] for _ts, gs in timeline]
                fh.write(json.dumps({"pk": g["game_pk"], "y": int(home_won),
                                     "s": states}) + "\n")
                written += 1
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                print(f"  {year}: {i}/{len(todo)}  ({rate:.1f} games/s, "
                      f"{(len(todo)-i)/rate/60:.0f} min left)", flush=True)
    print(f"{year}: wrote {written} games -> {path}", flush=True)
    return written


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2022, 2023, 2024, 2025]
    for y in years:
        fetch_season(y)

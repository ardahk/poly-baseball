from polybot.volatility import PriceHistory


def test_time_grid_volatility_is_stable_across_polling_density():
    sparse = PriceHistory()
    dense = PriceHistory()
    for ts, price in [(0, 0.50), (5, 0.55), (10, 0.50), (15, 0.60)]:
        sparse.add(price, ts)
    for ts in range(16):
        anchors = {0: 0.50, 5: 0.55, 10: 0.50, 15: 0.60}
        previous = max(k for k in anchors if k <= ts)
        dense.add(anchors[previous], ts)
    assert dense.realized_vol_time(15, 5) == sparse.realized_vol_time(15, 5)


def test_flips_within_uses_trailing_time_not_lifetime_count():
    history = PriceHistory(flip_band=0.03)
    for ts, price in [(0, 0.40), (10, 0.60), (100, 0.40), (110, 0.60)]:
        history.add(price, ts)
    assert history.flips == 3
    assert history.flips_within(20) == 2


def fill(history, prices, start=0.0, step=1.0):
    for i, p in enumerate(prices):
        history.add(p, ts=start + i * step)


def test_flip_counting_with_hysteresis():
    h = PriceHistory(flip_band=0.03)
    # crosses 0.5 decisively three times
    fill(h, [0.60, 0.55, 0.45, 0.40, 0.56, 0.60, 0.42])
    assert h.flips == 3


def test_jitter_around_half_does_not_flip():
    h = PriceHistory(flip_band=0.03)
    fill(h, [0.60, 0.51, 0.49, 0.51, 0.49, 0.52])
    assert h.flips == 0


def test_move_over_window():
    h = PriceHistory()
    fill(h, [0.50, 0.50, 0.50, 0.40], step=30.0)  # ts 0,30,60,90
    move = h.move(60.0)
    assert move is not None and abs(move - (-0.10)) < 1e-9


def test_move_none_without_enough_history():
    h = PriceHistory()
    fill(h, [0.5, 0.49], step=1.0)
    assert h.move(60.0) is None


def test_realized_vol_flat_vs_choppy():
    flat = PriceHistory()
    fill(flat, [0.5] * 20)
    choppy = PriceHistory()
    fill(choppy, [0.5 + (0.05 if i % 2 else -0.05) for i in range(20)])
    assert flat.realized_vol() == 0.0
    assert choppy.realized_vol() > 0.03


def test_playful_by_flips_or_vol():
    h = PriceHistory(flip_band=0.03)
    fill(h, [0.60, 0.40, 0.60])
    assert h.flips >= 2  # playful by regime crossings alone
    quiet = PriceHistory()
    fill(quiet, [0.60, 0.61, 0.60, 0.61])
    assert quiet.flips < 2 and quiet.realized_vol() < 0.05  # not playful

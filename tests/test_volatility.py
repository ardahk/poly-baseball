from polybot.volatility import PriceHistory


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
    assert h.is_playful(min_flips=2, min_volatility=99.0)
    quiet = PriceHistory()
    fill(quiet, [0.60, 0.61, 0.60, 0.61])
    assert not quiet.is_playful(min_flips=2, min_volatility=0.05)

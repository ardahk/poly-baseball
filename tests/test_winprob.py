from polybot.models import GameState
from polybot.winprob import home_win_probability


def gs(**kw):
    return GameState(game_pk=1, status="Live", **kw)


def test_tied_game_start_slightly_favors_home():
    p = home_win_probability(gs(inning=1, is_top=True))
    assert 0.5 < p < 0.58


def test_big_lead_late_is_near_certain():
    p = home_win_probability(gs(inning=9, is_top=True, outs=2,
                                home_score=8, away_score=1))
    assert p > 0.97


def test_lead_monotonic_in_score():
    probs = [home_win_probability(gs(inning=5, is_top=True, home_score=s))
             for s in range(0, 5)]
    assert all(a < b for a, b in zip(probs, probs[1:]))


def test_later_lead_worth_more():
    early = home_win_probability(gs(inning=2, is_top=True, home_score=2))
    late = home_win_probability(gs(inning=8, is_top=True, home_score=2))
    assert late > early


def test_bases_loaded_helps_batting_team():
    base = gs(inning=6, is_top=True, outs=1)
    loaded = gs(inning=6, is_top=True, outs=1,
                on_first=True, on_second=True, on_third=True)
    # away team batting with bases loaded -> home win prob drops
    assert home_win_probability(loaded) < home_win_probability(base)


def test_walkoff_position():
    p = home_win_probability(gs(inning=9, is_top=False, home_score=3, away_score=2))
    assert p == 1.0


def test_final_game():
    final = GameState(game_pk=1, status="Final", home_score=5, away_score=3)
    assert home_win_probability(final) == 1.0
    final.home_score, final.away_score = 3, 5
    assert home_win_probability(final) == 0.0


def test_probability_bounds():
    p = home_win_probability(gs(inning=9, is_top=True, outs=2,
                                home_score=20, away_score=0))
    assert 0.0 < p <= 0.999

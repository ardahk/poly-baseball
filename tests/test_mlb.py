from polybot.mlb import match_markets_to_games
from polybot.models import Market


def market(start_time=None):
    return Market(slug="aec-mlb-a-b", question="", home_team="Homers",
                  away_team="Awayers", long_team="Homers", start_time=start_time)


def game(pk, game_date):
    return {"game_pk": pk, "home": "Homers", "away": "Awayers",
            "status": "Final", "game_date": game_date}


def test_match_prefers_closest_start_time_in_series():
    # Same two teams play three days in a row (a series); the market must
    # attach to the game closest to its own start time, not the first hit.
    m = market(start_time=200_000.0)
    games = [game(1, 200_000.0 - 86_400), game(2, 200_000.0 + 300),
             game(3, 200_000.0 + 86_400)]
    match_markets_to_games([m], games)
    assert m.game_pk == 2


def test_match_rejects_games_outside_window():
    m = market(start_time=200_000.0)
    match_markets_to_games([m], [game(1, 200_000.0 + 30 * 3600)])
    assert m.game_pk is None

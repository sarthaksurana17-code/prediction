"""
Utility classes and functions used across the prediction model.
"""

import pandas as pd

# Static, approximate FIFA-style world ranking (lower = stronger) as of mid-2026.
# This is a hand-maintained approximation for teams that appear in our dataset,
# not a live feed from FIFA. For production use, replace with a call to FIFA's
# official rankings API/CSV.
_FIFA_RANKINGS = {
    "Argentina": 1, "France": 2, "Spain": 3, "England": 4, "Belgium": 5,
    "Brazil": 6, "Portugal": 7, "Netherlands": 8, "Germany": 9, "Croatia": 10,
    "Morocco": 12, "Italy": 13, "Colombia": 14, "Switzerland": 15, "USA": 16,
    "Denmark": 17, "Mexico": 18, "Uruguay": 19, "Senegal": 20, "Japan": 21,
    "Norway": 24, "Sweden": 25, "Poland": 26, "Iran": 27, "South Korea": 28,
    "Ecuador": 29, "Egypt": 30, "Wales": 31, "Canada": 32, "Austria": 33,
    "Ukraine": 34, "Serbia": 35, "Tunisia": 36, "Australia": 37, "Ghana": 38,
    "Qatar": 39, "Costa Rica": 40, "Saudi Arabia": 41, "Cabo Verde": 42,
    "Ivory Coast": 43, "Algeria": 44, "Nigeria": 45, "Panama": 46,
    "Paraguay": 47, "Bosnia and Herzegovina": 48, "Iraq": 49, "New Zealand": 50,
    "Russia": 51, "Chile": 52, "Honduras": 53, "South Africa": 54,
    "DR Congo": 55, "Czechia": 56,
}
_DEFAULT_RANKING = 60  # fallback for any team not in the dict above


def get_fifa_ranking(team: str) -> int:
    """Return an approximate FIFA world ranking for a team (lower = stronger)."""
    return _FIFA_RANKINGS.get(team, _DEFAULT_RANKING)


class TeamStats:
    """Computes team strength metrics from a historical_matches DataFrame."""

    def __init__(self, historical_matches: pd.DataFrame):
        self.matches = historical_matches

    def _team_matches(self, team: str) -> pd.DataFrame:
        return self.matches[
            (self.matches["home_team"] == team) | (self.matches["away_team"] == team)
        ]

    def get_team_strength(self, team: str) -> float:
        """
        Team strength score (0-10 scale) derived from historical win rate and
        average goal difference. Teams with no match history (e.g. Norway prior
        to 2026, which last played a men's WC in 1998) fall back to a rating
        derived purely from FIFA ranking.
        """
        team_matches = self._team_matches(team)

        if len(team_matches) == 0:
            # No history: derive a baseline purely from FIFA ranking.
            rank = get_fifa_ranking(team)
            return max(1.0, 8.0 - (rank / 15))

        points = 0
        goal_diff_total = 0
        for _, row in team_matches.iterrows():
            is_home = row["home_team"] == team
            gf = row["home_score"] if is_home else row["away_score"]
            ga = row["away_score"] if is_home else row["home_score"]
            goal_diff_total += (gf - ga)
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1

        avg_points = points / len(team_matches)
        avg_gd = goal_diff_total / len(team_matches)

        # Scale to a roughly 0-10 range: 3 pts/game (always winning) -> ~10
        strength = (avg_points / 3.0) * 7.0 + max(-2, min(2, avg_gd)) * 0.75
        return round(max(1.0, min(10.0, strength)), 2)

    def calculate_goals_for_against(self, team: str):
        """Returns (avg_goals_for, avg_goals_against) for a team's match history."""
        team_matches = self._team_matches(team)

        if len(team_matches) == 0:
            return 1.2, 1.2  # neutral fallback for teams with no recorded history

        gf_total, ga_total = 0, 0
        for _, row in team_matches.iterrows():
            is_home = row["home_team"] == team
            gf_total += row["home_score"] if is_home else row["away_score"]
            ga_total += row["away_score"] if is_home else row["home_score"]

        n = len(team_matches)
        return round(gf_total / n, 2), round(ga_total / n, 2)


class PlayerStats:
    """Computes player-level scoring metrics from a player_stats DataFrame."""

    def __init__(self, players_df: pd.DataFrame):
        self.players = players_df

    def _get_player_row(self, player_name: str):
        row = self.players[self.players["player_name"] == player_name]
        return row.iloc[0] if len(row) > 0 else None

    def get_player_scoring_rate(self, player_name: str) -> float:
        """Career goals-per-game rate. Falls back to 0 if the player isn't found."""
        row = self._get_player_row(player_name)
        if row is None or row["games_played"] == 0:
            return 0.0
        return round(row["goals"] / row["games_played"], 4)

    def get_player_rating(self, player_name: str) -> float:
        row = self._get_player_row(player_name)
        return float(row["rating"]) if row is not None else 6.5

    def calculate_injury_factor(self, player_name: str) -> float:
        """
        Fitness/availability multiplier. We don't have real injury-report data,
        so this returns a fixed 1.0 (fully fit) for every player rather than
        fabricating an injury status. Wire this up to a real injury feed if
        one becomes available.
        """
        return 1.0

    def get_current_tournament_rate(self, player_name: str) -> float:
        """Goals-per-game rate for the current (2026) tournament specifically,
        using the goals_2026_wc / games_2026_wc columns if present."""
        row = self._get_player_row(player_name)
        if row is None or "goals_2026_wc" not in row or row.get("games_2026_wc", 0) == 0:
            return 0.0
        return round(row["goals_2026_wc"] / row["games_2026_wc"], 4)

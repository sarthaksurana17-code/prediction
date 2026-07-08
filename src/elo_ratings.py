"""
elo_ratings.py

Elo rating system for national football teams, adapted from the World Football
Elo Ratings methodology (eloratings.net). Drop this into src/ and use it as
a feature source for TournamentPredictor / MatchOutcomeModel instead of, or
alongside, static historical averages.

Why this matters: static "team strength from historical averages" doesn't
capture current form. A team that was dominant in 2014 but declined by 2022
gets over-weighted. Elo updates after every match, so it reflects recent form,
margin of victory, and match importance -- much closer to how real forecasting
models (FiveThirtyEight SPI, eloratings.net) actually work.

Usage:
    from src.elo_ratings import EloRatingSystem

    elo = EloRatingSystem()
    elo.process_match_history(historical_matches_df)  # builds ratings from past results
    rating = elo.get_rating("Argentina")
    win_prob = elo.win_probability("Argentina", "Mexico", neutral_venue=True)
"""

import math
from collections import defaultdict


# K-factor by competition importance (higher = ratings move faster after result)
# Values follow the eloratings.net convention.
K_FACTOR = {
    "friendly": 20,
    "qualifier": 30,
    "continental_final": 40,   # e.g. Copa America / Euro final stage
    "world_cup_group": 50,
    "world_cup_knockout": 60,
    "world_cup_final": 65,
}

DEFAULT_RATING = 1500
HOME_ADVANTAGE = 65  # Elo points added to home team in non-neutral matches


class EloRatingSystem:
    def __init__(self, default_rating: float = DEFAULT_RATING):
        self.default_rating = default_rating
        self.ratings = defaultdict(lambda: default_rating)
        self.history = []  # list of dicts: date, team, rating, opponent, result

    def get_rating(self, team: str) -> float:
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability team A beats team B (no draw handling here; see win_probability)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def win_probability(self, team_a: str, team_b: str, neutral_venue: bool = True):
        """
        Returns (p_win_a, p_draw, p_win_b) using a standard football-Elo-to-outcome
        conversion. Draws are estimated from the rating gap: closer ratings -> higher
        draw probability, based on empirical calibration used by eloratings.net-style
        models (draw probability peaks near ~28% for evenly matched teams).
        """
        ra = self.ratings[team_a] + (0 if neutral_venue else HOME_ADVANTAGE)
        rb = self.ratings[team_b]
        diff = ra - rb

        # Base win expectancy (Elo logistic)
        p_a_beats_b_no_draw = self.expected_score(ra, rb)

        # Draw probability model: peaks when teams are evenly matched, shrinks as
        # the gap widens. Coefficients tuned to roughly match historical WC draw rates (~24%).
        draw_prob = 0.28 * math.exp(-((diff / 200.0) ** 2))

        p_win_a = p_a_beats_b_no_draw * (1 - draw_prob)
        p_win_b = (1 - p_a_beats_b_no_draw) * (1 - draw_prob)

        return round(p_win_a, 4), round(draw_prob, 4), round(p_win_b, 4)

    def update_ratings(self, team_a, team_b, score_a, score_b, competition="friendly",
                        neutral_venue=True, date=None):
        """
        Update Elo ratings after a single match result.
        score_a, score_b: goals scored by each team.
        """
        ra = self.ratings[team_a] + (0 if neutral_venue else HOME_ADVANTAGE)
        rb = self.ratings[team_b]

        expected_a = self.expected_score(ra, rb)

        if score_a > score_b:
            actual_a = 1.0
        elif score_a < score_b:
            actual_a = 0.0
        else:
            actual_a = 0.5

        # Goal-difference multiplier (bigger wins move ratings more, diminishing returns)
        goal_diff = abs(score_a - score_b)
        gd_multiplier = 1.0 if goal_diff <= 1 else (1.5 if goal_diff == 2 else (1.75 + (goal_diff - 3) / 8))

        k = K_FACTOR.get(competition, 20) * gd_multiplier

        delta = k * (actual_a - expected_a)
        self.ratings[team_a] += delta
        self.ratings[team_b] -= delta

        self.history.append({
            "date": date, "team_a": team_a, "team_b": team_b,
            "score_a": score_a, "score_b": score_b,
            "rating_a_after": self.ratings[team_a], "rating_b_after": self.ratings[team_b],
        })

    def process_match_history(self, matches_df, date_col="date", home_col="home_team",
                               away_col="away_team", home_score_col="home_score",
                               away_score_col="away_score", competition_col=None):
        """
        Feed in historical_matches.csv (as a pandas DataFrame), sorted oldest to newest,
        to build up ratings before the tournament starts. Adjust column names to match
        your actual CSV schema.
        """
        df = matches_df.sort_values(date_col)
        for _, row in df.iterrows():
            competition = row[competition_col] if competition_col else "friendly"
            self.update_ratings(
                row[home_col], row[away_col],
                row[home_score_col], row[away_score_col],
                competition=competition, neutral_venue=False, date=row[date_col],
            )

    def rankings(self, top_n=20):
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]

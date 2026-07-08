"""
Tournament Predictor - World Cup 2026

Unlike a from-scratch group-stage simulation, this predicts forward from the
ACTUAL current state of the tournament (quarterfinals, July 8 2026), using
Elo ratings built from real historical results (2014/2018/2022 + this
tournament's results to date).

Method: Monte Carlo simulation. We simulate the remaining bracket (QF -> SF ->
Final) 20,000 times using Elo-derived win/draw/loss probabilities (draws are
resolved via penalties, 50/50, matching real knockout rules) and report the
champion frequency distribution. This gives a genuine probability, not a
false-certainty single answer -- which is the honest way to report the output
of a statistical model on a live, not-yet-decided tournament.
"""

import random
from collections import Counter

from src.data_loader import DataLoader
from src.elo_ratings import EloRatingSystem


# Real quarterfinal bracket as of July 8, 2026
QUARTERFINALS = [
    ("France", "Morocco"),
    ("Spain", "Belgium"),
    ("Norway", "England"),
    ("Argentina", "Switzerland"),
]

# Semifinal pairings follow the bracket: winner(QF1) vs winner(QF2), winner(QF3) vs winner(QF4)
N_SIMULATIONS = 20000


class TournamentPredictor:
    def __init__(self):
        self.loader = DataLoader()
        self.data = self.loader.get_all_data()
        self.elo = EloRatingSystem()
        self.elo.process_match_history(
            self.data["historical_matches"], competition_col="competition"
        )

    def _simulate_match(self, team_a, team_b) -> str:
        """Simulate a single knockout match, returns the winner. Draws go to a coin flip
        (approximating penalty shootout unpredictability, since Elo has no signal there)."""
        p_a, p_draw, p_b = self.elo.win_probability(team_a, team_b, neutral_venue=True)
        roll = random.random()
        if roll < p_a:
            return team_a
        elif roll < p_a + p_draw:
            return team_a if random.random() < 0.5 else team_b
        else:
            return team_b

    def _simulate_bracket_once(self):
        sf_pairs = []
        for a, b in QUARTERFINALS:
            sf_pairs.append(self._simulate_match(a, b))

        finalists = [
            self._simulate_match(sf_pairs[0], sf_pairs[1]),
            self._simulate_match(sf_pairs[2], sf_pairs[3]),
        ]
        champion = self._simulate_match(finalists[0], finalists[1])
        return champion, sf_pairs, finalists

    def run_full_tournament_simulation(self, n_simulations: int = N_SIMULATIONS):
        print("\n" + "#" * 70)
        print("TOURNAMENT SIMULATION - MONTE CARLO (from current QF stage)".center(70))
        print("#" * 70)

        print("\nCurrent Elo ratings of remaining teams:")
        remaining_teams = [t for pair in QUARTERFINALS for t in pair]
        for team in remaining_teams:
            print(f"  {team:15} {self.elo.get_rating(team):.0f}")

        print(f"\nQuarterfinal matchups:")
        for a, b in QUARTERFINALS:
            p_a, p_draw, p_b = self.elo.win_probability(a, b)
            print(f"  {a} vs {b}: {a} {p_a:.0%} / Draw {p_draw:.0%} / {b} {p_b:.0%}")

        champion_counts = Counter()
        finalist_counts = Counter()
        semifinalist_counts = Counter()

        for _ in range(n_simulations):
            champion, sf_pairs, finalists = self._simulate_bracket_once()
            champion_counts[champion] += 1
            for f in finalists:
                finalist_counts[f] += 1
            for s in sf_pairs:
                semifinalist_counts[s] += 1

        print(f"\n" + "=" * 70)
        print(f"CHAMPION PROBABILITY (based on {n_simulations:,} simulations)")
        print("=" * 70)
        for team, count in champion_counts.most_common():
            print(f"  {team:15} {count / n_simulations:.1%}")

        champion = champion_counts.most_common(1)[0][0]
        champion_prob = champion_counts[champion] / n_simulations

        print(f"\n🏆 MODEL PREDICTION: {champion} ({champion_prob:.1%} win probability)")
        print("   (This is a probabilistic estimate from an Elo-based Monte Carlo")
        print("    simulation, not a certainty -- the tournament is still live.)")

        self.last_semifinalist_avg_games = self._estimate_avg_remaining_games(
            semifinalist_counts, finalist_counts, champion_counts, n_simulations
        )

        return champion

    def _estimate_avg_remaining_games(self, semifinalist_counts, finalist_counts,
                                       champion_counts, n_simulations):
        """For each of the 8 QF teams, estimate expected additional matches played
        (1 = lost in QF, 2 = lost in SF, 3 = lost in Final, 3 = won it -- final is also
        their 3rd remaining game). Used by the Golden Boot projection."""
        avg_games = {}
        for a, b in QUARTERFINALS:
            for team in (a, b):
                reached_sf = semifinalist_counts.get(team, 0) / n_simulations
                reached_final = finalist_counts.get(team, 0) / n_simulations
                # Expected additional games = at least 1 (the QF itself) + prob of SF + prob of Final
                avg_games[team] = 1 + reached_sf + reached_final
        return avg_games

    def predict_golden_boot(self, top_n=8):
        """
        Golden Boot projection grounded in REAL current tournament data:
        final_projected_goals = goals already scored (verified, as of July 7-8 2026)
                                 + (this-tournament per-game scoring rate * expected
                                    additional games remaining, from the Monte Carlo
                                    bracket simulation)
        """
        if not hasattr(self, "last_semifinalist_avg_games"):
            self.run_full_tournament_simulation()

        players_df = self.data["player_stats"]
        avg_games = self.last_semifinalist_avg_games

        projections = []
        for _, p in players_df.iterrows():
            team = p["team"]
            current_goals = p.get("goals_2026_wc", 0)
            current_games = p.get("games_2026_wc", 5)
            per_game_rate = current_goals / current_games if current_games > 0 else 0

            expected_remaining_games = avg_games.get(team, 0)
            projected_additional = per_game_rate * expected_remaining_games
            final_projection = current_goals + projected_additional

            projections.append({
                "player": p["player_name"],
                "team": team,
                "goals_so_far": current_goals,
                "expected_remaining_games": round(expected_remaining_games, 2),
                "projected_final_goals": round(final_projection, 2),
            })

        projections.sort(key=lambda x: x["projected_final_goals"], reverse=True)

        print("\n" + "=" * 70)
        print("GOLDEN BOOT PROJECTION (grounded in verified goals through July 7-8, 2026)")
        print("=" * 70)
        print(f"{'Player':<20}{'Team':<15}{'Goals so far':<15}{'Projected final':<15}")
        print("-" * 70)
        for row in projections[:top_n]:
            print(f"{row['player']:<20}{row['team']:<15}{row['goals_so_far']:<15}"
                  f"{row['projected_final_goals']:<15.2f}")

        leader = projections[0]
        print(f"\n🥇 PROJECTED GOLDEN BOOT: {leader['player']} ({leader['team']}), "
              f"~{leader['projected_final_goals']:.1f} projected goals")

        return projections[:top_n]


if __name__ == "__main__":
    predictor = TournamentPredictor()
    champion = predictor.run_full_tournament_simulation()
    predictor.predict_golden_boot()

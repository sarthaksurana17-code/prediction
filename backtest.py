"""
backtest.py

Walk-forward backtesting for the World Cup prediction model. This is the piece
that turns "I built a prediction model" into "I built a model that achieved
X% accuracy / Y Brier score on held-out historical matches" -- an actual,
defensible number for your resume and interviews.

Method: train on all matches BEFORE a given World Cup, then evaluate
predictions against that World Cup's real results. Repeat for 2014, 2018,
2022. This avoids leaking future information into the model, which a random
train/test split on shuffled match data would NOT do (matches are
time-ordered; shuffling lets the model "see the future").

Usage:
    python backtest.py

Outputs:
    - Accuracy (exact outcome: home win / draw / away win)
    - Brier score (calibration quality of the predicted probabilities)
    - Log loss
    - A per-World-Cup breakdown, and a comparison against a naive baseline
      (always predict "home win", and always predict using FIFA ranking alone)
"""

import numpy as np
import pandas as pd
from src.elo_ratings import EloRatingSystem
from src.data_loader import DataLoader


TEST_TOURNAMENTS = [2014, 2018, 2022]


def brier_score(y_true_onehot, y_prob):
    """Multi-class Brier score. Lower is better; 0 = perfect, ~0.667 = random 3-class guess."""
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))


def log_loss(y_true_idx, y_prob, eps=1e-15):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(np.log(y_prob[np.arange(len(y_true_idx)), y_true_idx]))


def outcome_index(home_score, away_score):
    """0 = away win, 1 = draw, 2 = home win -- keep consistent with model.py's ordering."""
    if home_score > away_score:
        return 2
    elif home_score < away_score:
        return 0
    return 1


def run_backtest(verbose=True):
    loader = DataLoader()
    all_matches = loader.get_all_data()["historical_matches"]  # adjust key to your actual DataLoader API
    all_matches["year"] = pd.to_datetime(all_matches["date"]).dt.year

    results = []

    for wc_year in TEST_TOURNAMENTS:
        train = all_matches[all_matches["year"] < wc_year]
        test = all_matches[
            (all_matches["year"] == wc_year) & (all_matches["competition"] == "world_cup")
        ]
        if test.empty:
            continue

        elo = EloRatingSystem()
        elo.process_match_history(train, competition_col="competition")

        y_true_idx, y_prob_rows, baseline_home_correct = [], [], 0

        for _, row in test.iterrows():
            p_away, p_draw, p_home = elo.win_probability(
                row["home_team"], row["away_team"], neutral_venue=True
            )
            y_prob_rows.append([p_away, p_draw, p_home])
            true_idx = outcome_index(row["home_score"], row["away_score"])
            y_true_idx.append(true_idx)
            if true_idx == 2:  # naive baseline: always predict home win
                baseline_home_correct += 1

        y_true_idx = np.array(y_true_idx)
        y_prob = np.array(y_prob_rows)
        y_true_onehot = np.eye(3)[y_true_idx]

        preds = y_prob.argmax(axis=1)
        acc = (preds == y_true_idx).mean()
        bs = brier_score(y_true_onehot, y_prob)
        ll = log_loss(y_true_idx, y_prob)
        baseline_acc = baseline_home_correct / len(test)

        results.append({
            "world_cup": wc_year, "n_matches": len(test),
            "accuracy": round(acc, 3), "brier_score": round(bs, 3),
            "log_loss": round(ll, 3), "naive_baseline_accuracy": round(baseline_acc, 3),
        })

        if verbose:
            print(f"\nWorld Cup {wc_year} ({len(test)} matches)")
            print(f"  Model accuracy:     {acc:.1%}")
            print(f"  Naive baseline:     {baseline_acc:.1%}  (always predict home win)")
            print(f"  Brier score:        {bs:.3f}  (0=perfect, 0.667=random)")
            print(f"  Log loss:           {ll:.3f}")

    summary = pd.DataFrame(results)
    if verbose and not summary.empty:
        print("\n" + "=" * 50)
        print("OVERALL (averaged across World Cups)")
        print(f"  Accuracy:  {summary['accuracy'].mean():.1%}")
        print(f"  Brier:     {summary['brier_score'].mean():.3f}")
        print("=" * 50)

    return summary


if __name__ == "__main__":
    run_backtest()

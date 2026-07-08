"""
Data loading module. Reads historical_matches.csv and player_stats.csv from
the repo's data/ directory (resolved relative to this file, so it works
regardless of the current working directory the script is run from).
"""

import os
import pandas as pd


class DataLoader:
    def __init__(self):
        src_dir = os.path.dirname(os.path.abspath(__file__))
        self.repo_root = os.path.dirname(src_dir)
        self.data_dir = os.path.join(self.repo_root, "data")

        # Real World Cup 2026 quarterfinal groupings, current as of July 8, 2026.
        # Used by TournamentPredictor instead of a fictional group-stage draw,
        # since the actual tournament has already progressed past the group stage.
        self.groups = {
            "QF1": ["France", "Morocco"],
            "QF2": ["Spain", "Belgium"],
            "QF3": ["Norway", "England"],
            "QF4": ["Argentina", "Switzerland"],
        }

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected data file not found: {path}\n"
                f"Make sure {filename} exists in the data/ directory."
            )
        return pd.read_csv(path)

    def get_all_data(self) -> dict:
        return {
            "historical_matches": self._load_csv("historical_matches.csv"),
            "player_stats": self._load_csv("player_stats.csv"),
        }

"""
Top Goalscorer Prediction Module
"""

import pandas as pd
import numpy as np
from src.utils import PlayerStats, get_fifa_ranking
from src.data_loader import DataLoader


class GoalscorerPredictor:
    """Predict top goalscorer for the tournament"""
    
    def __init__(self):
        self.loader = DataLoader()
        self.data = self.loader.get_all_data()
        self.player_stats = PlayerStats(self.data['player_stats'])
        self.players_df = self.data['player_stats']
    
    def calculate_scoring_potential(self, player_row):
        """Calculate scoring potential score for a player"""
        
        # Get player statistics
        scoring_rate = self.player_stats.get_player_scoring_rate(player_row['player_name'])
        rating = self.player_stats.get_player_rating(player_row['player_name'])
        
        # Get team strength (better teams advance further)
        team = player_row['team']
        team_data = pd.DataFrame([{'team': team}])
        
        # Team strength factor (0-1)
        fifa_rank = get_fifa_ranking(team)
        team_strength_factor = 1 - (fifa_rank / 100)
        
        # Age factor (peak years 26-32)
        age = player_row['age']
        if 26 <= age <= 32:
            age_factor = 1.0
        elif age < 26:
            age_factor = 0.95 + (age / 100)
        else:
            age_factor = max(0.7, 1.0 - ((age - 32) * 0.05))
        
        # Fitness factor
        fitness_factor = self.player_stats.calculate_injury_factor(player_row['player_name'])
        
        # Calculate overall scoring potential
        scoring_potential = (
            scoring_rate * 3 +  # Weight scoring rate
            (rating / 10) * 1 +  # Rating
            team_strength_factor * 2 +  # Team strength
            age_factor * 1 +  # Age
            fitness_factor * 1  # Fitness
        )
        
        return scoring_potential
    
    def estimate_games_played(self, player_row, is_champion=False):
        """Estimate how many games a player will play in the tournament"""
        
        team = player_row['team']
        fifa_rank = get_fifa_ranking(team)
        
        # Base games (group stage + knockouts)
        base_games = 3  # Group stage minimum
        
        # Advancement probability based on FIFA ranking
        if fifa_rank <= 10:
            prob_ro32 = 0.95
            prob_qf = 0.70
            prob_sf = 0.40
            prob_final = 0.20
        elif fifa_rank <= 25:
            prob_ro32 = 0.85
            prob_qf = 0.50
            prob_sf = 0.25
            prob_final = 0.10
        else:
            prob_ro32 = 0.65
            prob_qf = 0.30
            prob_sf = 0.10
            prob_final = 0.02
        
        expected_games = base_games
        expected_games += prob_ro32 * 1
        expected_games += prob_qf * 1
        expected_games += prob_sf * 1
        expected_games += prob_final * 1
        
        if is_champion:
            expected_games = 7  # All matches including final
        
        return expected_games
    
    def predict_goals(self, scoring_potential, games_played, scoring_rate):
        """Predict number of goals based on potential and games"""
        
        # Use Poisson-like distribution
        expected_goals = scoring_potential * games_played * scoring_rate
        
        # Add some variance (normal distribution)
        predicted_goals = np.random.normal(expected_goals, expected_goals * 0.2)
        
        return max(0, int(np.round(predicted_goals)))
    
    def predict_top_scorers(self, top_n=10):
        """Predict top goalscorers for the tournament"""
        
        print("\n" + "="*70)
        print("TOP GOALSCORER PREDICTION - WORLD CUP 2026")
        print("="*70 + "\n")
        
        # Calculate scoring potential for all players
        predictions = []
        
        for idx, player in self.players_df.iterrows():
            player_name = player['player_name']
            team = player['team']
            position = player['position']
            
            # Skip non-forwards (simplified)
            if position not in ['Forward', 'Striker']:
                continue
            
            scoring_potential = self.calculate_scoring_potential(player)
            scoring_rate = self.player_stats.get_player_scoring_rate(player_name)
            games_played = self.estimate_games_played(player)
            predicted_goals = self.predict_goals(scoring_potential, games_played, scoring_rate)
            
            predictions.append({
                'player': player_name,
                'team': team,
                'predicted_goals': predicted_goals,
                'scoring_potential': scoring_potential,
                'games_played': games_played,
                'scoring_rate': scoring_rate
            })
        
        # Sort by predicted goals
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('predicted_goals', ascending=False)
        
        # Display top scorers
        print(f"{'Rank':<6}{'Player':<25}{'Team':<15}{'Goals':<8}{'Potential':<12}")
        print("-" * 70)
        
        for rank, (idx, player) in enumerate(predictions_df.head(top_n).iterrows(), 1):
            print(f"{rank:<6}{player['player']:<25}{player['team']:<15}"
                  f"{player['predicted_goals']:<8}{player['scoring_potential']:.2f}")
        
        # Highlight top goalscorer
        top_scorer = predictions_df.iloc[0]
        print(f"\n{'🥇 TOP GOALSCORER PREDICTION'.center(70)}")
        print("-" * 70)
        print(f"Player: {top_scorer['player']}")
        print(f"Team: {top_scorer['team']}")
        print(f"Predicted Goals: {top_scorer['predicted_goals']}")
        print(f"Confidence: {min(100, top_scorer['scoring_potential'] * 15):.1f}%")
        
        return predictions_df.head(top_n)
    
    def predict_player_tournament_stats(self, player_name):
        """Predict detailed stats for a specific player"""
        
        player = self.players_df[self.players_df['player_name'] == player_name]
        
        if len(player) == 0:
            print(f"Player {player_name} not found in database")
            return None
        
        player = player.iloc[0]
        
        print("\n" + "="*50)
        print(f"TOURNAMENT PREDICTION: {player_name}")
        print("="*50)
        
        scoring_potential = self.calculate_scoring_potential(player)
        scoring_rate = self.player_stats.get_player_scoring_rate(player_name)
        games_played = self.estimate_games_played(player)
        predicted_goals = self.predict_goals(scoring_potential, games_played, scoring_rate)
        
        print(f"Team: {player['team']}")
        print(f"Position: {player['position']}")
        print(f"Age: {player['age']}")
        print(f"\nPredicted Stats:")
        print(f"  Games Played: {games_played:.1f}")
        print(f"  Goals: {predicted_goals}")
        print(f"  Assists (Est): {int(games_played * 0.5)}")
        print(f"  Shots on Target (Est): {int(predicted_goals * 2.5)}")
        print(f"  Scoring Potential Score: {scoring_potential:.2f}/10")
        
        return {
            'player': player_name,
            'team': player['team'],
            'goals': predicted_goals,
            'games': games_played,
            'potential': scoring_potential
        }


if __name__ == "__main__":
    predictor = GoalscorerPredictor()
    top_scorers = predictor.predict_top_scorers(top_n=10)
    
    # Detailed prediction for top player
    if len(top_scorers) > 0:
        top_player = top_scorers.iloc[0]['player']
        predictor.predict_player_tournament_stats(top_player)

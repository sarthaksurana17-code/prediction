"""
Golden Ball Winner (Best Player) Prediction Module
"""

import pandas as pd
import numpy as np
from src.utils import get_fifa_ranking
from src.data_loader import DataLoader


class GoldenBallPredictor:
    """Predict Golden Ball (Best Player) winner"""
    
    def __init__(self):
        self.loader = DataLoader()
        self.data = self.loader.get_all_data()
        self.players_df = self.data['player_stats']
    
    def calculate_golden_ball_score(self, player_row):
        """
        Calculate Golden Ball potential score
        Factors:
        - Overall player rating
        - Consistent performance (games played)
        - Goals and assists potential
        - Team advancement (players in winning teams score higher)
        - Age and experience
        """
        
        # Base rating (50% weight)
        rating_score = player_row['rating'] * 5
        
        # Experience factor
        games = player_row['games_played']
        experience_score = min(10, games / 20)
        
        # Scoring/assist potential (20% weight)
        goals_per_game = player_row['goals'] / games if games > 0 else 0
        scoring_score = goals_per_game * 10
        
        # Leadership and clutch factor (based on age and position)
        age = player_row['age']
        if 26 <= age <= 34:
            leadership_score = 8
        elif age > 34:
            leadership_score = 6
        else:
            leadership_score = 7
        
        # Team quality factor
        team = player_row['team']
        fifa_rank = get_fifa_ranking(team)
        team_factor = (100 - fifa_rank) / 10
        
        # Overall golden ball score (0-100)
        gb_score = (
            rating_score * 0.35 +
            experience_score * 0.15 +
            scoring_score * 0.20 +
            leadership_score * 0.15 +
            team_factor * 0.15
        )
        
        return gb_score
    
    def calculate_tournament_performance_multiplier(self, player_row):
        """Calculate how well player is likely to perform in this specific tournament"""
        
        # Recent form (assume recent goals = good form)
        recent_goals = player_row['goals']  # Latest data
        form_multiplier = min(1.2, 0.8 + (recent_goals / 100))
        
        # Pressure/big match experience
        rating = player_row['rating']
        if rating >= 8.3:
            pressure_multiplier = 1.15
        elif rating >= 8.0:
            pressure_multiplier = 1.10
        elif rating >= 7.5:
            pressure_multiplier = 1.05
        else:
            pressure_multiplier = 1.0
        
        total_multiplier = form_multiplier * pressure_multiplier
        return total_multiplier
    
    def predict_winner(self, top_n=5):
        """Predict Golden Ball winner"""
        
        print("\n" + "="*70)
        print("GOLDEN BALL PREDICTION - WORLD CUP 2026")
        print("="*70 + "\n")
        
        predictions = []
        
        for idx, player in self.players_df.iterrows():
            player_name = player['player_name']
            team = player['team']
            
            # Calculate base score
            gb_score = self.calculate_golden_ball_score(player)
            
            # Apply tournament multiplier
            multiplier = self.calculate_tournament_performance_multiplier(player)
            final_score = gb_score * multiplier
            
            predictions.append({
                'player': player_name,
                'team': team,
                'rating': player['rating'],
                'base_score': gb_score,
                'tournament_multiplier': multiplier,
                'final_score': final_score,
                'age': player['age']
            })
        
        # Sort by final score
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('final_score', ascending=False)
        
        # Display predictions
        print(f"{'Rank':<6}{'Player':<25}{'Team':<15}{'Rating':<8}{'Score':<10}")
        print("-" * 70)
        
        for rank, (idx, player) in enumerate(predictions_df.head(top_n).iterrows(), 1):
            print(f"{rank:<6}{player['player']:<25}{player['team']:<15}"
                  f"{player['rating']:<8.1f}{player['final_score']:<10.2f}")
        
        # Highlight winner
        winner = predictions_df.iloc[0]
        print(f"\n{'🏆 GOLDEN BALL PREDICTION'.center(70)}")
        print("-" * 70)
        print(f"Player: {winner['player']}")
        print(f"Team: {winner['team']}")
        print(f"Rating: {winner['rating']:.1f}/10")
        print(f"Final Score: {winner['final_score']:.2f}")
        print(f"Confidence: {min(100, winner['final_score'] * 8):.1f}%")
        
        return predictions_df.head(top_n)
    
    def predict_other_awards(self):
        """Predict other tournament awards"""
        
        print("\n" + "="*70)
        print("OTHER AWARD PREDICTIONS")
        print("="*70 + "\n")
        
        # Silver Ball (2nd best player)
        predictions_df = self.predict_winner(top_n=3)
        
        if len(predictions_df) > 1:
            silver = predictions_df.iloc[1]
            print(f"\n🥈 SILVER BALL: {silver['player']} ({silver['team']})")
        
        if len(predictions_df) > 2:
            bronze = predictions_df.iloc[2]
            print(f"🥉 BRONZE BALL: {bronze['player']} ({bronze['team']})")
        
        # Best Goalkeeper (simplified - pick highest rated goalkeeper)
        print(f"\n🧤 BEST GOALKEEPER (TBD - requires goalkeeper data)")
        print(f"🎯 BEST YOUNG PLAYER: (Youngest top performer TBD)")
        
        return predictions_df
    
    def analyze_player_for_award(self, player_name):
        """Analyze specific player's chances for Golden Ball"""
        
        player = self.players_df[self.players_df['player_name'] == player_name]
        
        if len(player) == 0:
            print(f"Player {player_name} not found")
            return None
        
        player = player.iloc[0]
        
        print("\n" + "="*60)
        print(f"GOLDEN BALL ANALYSIS: {player_name}")
        print("="*60)
        
        gb_score = self.calculate_golden_ball_score(player)
        multiplier = self.calculate_tournament_performance_multiplier(player)
        final_score = gb_score * multiplier
        
        print(f"\nPlayer Profile:")
        print(f"  Team: {player['team']}")
        print(f"  Position: {player['position']}")
        print(f"  Age: {player['age']}")
        print(f"  Rating: {player['rating']:.1f}/10")
        
        print(f"\nScoring Analysis:")
        print(f"  Total Goals: {player['goals']}")
        print(f"  Games Played: {player['games_played']}")
        print(f"  Goals per Game: {player['goals']/player['games_played']:.2f}")
        
        print(f"\nGolden Ball Score Breakdown:")
        print(f"  Base Score: {gb_score:.2f}")
        print(f"  Tournament Multiplier: {multiplier:.2f}x")
        print(f"  Final Score: {final_score:.2f}/100")
        
        # Determine likelihood
        if final_score >= 80:
            likelihood = "VERY HIGH 🌟"
        elif final_score >= 70:
            likelihood = "HIGH 👍"
        elif final_score >= 60:
            likelihood = "MODERATE ➜"
        elif final_score >= 50:
            likelihood = "POSSIBLE 📊"
        else:
            likelihood = "LOW"
        
        print(f"\nGolden Ball Likelihood: {likelihood}")
        
        return {
            'player': player_name,
            'score': final_score,
            'rating': player['rating'],
            'team': player['team']
        }


if __name__ == "__main__":
    predictor = GoldenBallPredictor()
    
    # Overall predictions
    top_candidates = predictor.predict_winner(top_n=5)
    
    # Other awards
    predictor.predict_other_awards()
    
    # Detailed analysis for top player
    if len(top_candidates) > 0:
        top_player = top_candidates.iloc[0]['player']
        predictor.analyze_player_for_award(top_player)

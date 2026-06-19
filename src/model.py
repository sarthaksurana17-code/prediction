"""
Machine Learning Model Training Module
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

from src.data_loader import DataLoader, prepare_training_data
from src.utils import TeamStats, get_fifa_ranking


class MatchOutcomeModel:
    """Train and use ML model for match outcome prediction"""
    
    def __init__(self):
        self.loader = DataLoader()
        self.data = self.loader.get_all_data()
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'models/match_outcome_model.pkl'
    
    def create_features(self, home_team, away_team, historical_matches):
        """Create feature vector for a match"""
        
        team_stats = TeamStats(historical_matches)
        
        # Team strengths
        home_strength = team_stats.get_team_strength(home_team)
        away_strength = team_stats.get_team_strength(away_team)
        
        # Goals for/against
        home_gf, home_ga = team_stats.calculate_goals_for_against(home_team)
        away_gf, away_ga = team_stats.calculate_goals_for_against(away_team)
        
        # FIFA rankings
        home_fifa = get_fifa_ranking(home_team)
        away_fifa = get_fifa_ranking(away_team)
        
        # Home advantage
        home_advantage = 1
        
        features = [
            home_strength,
            away_strength,
            home_strength - away_strength,
            home_gf,
            home_ga,
            away_gf,
            away_ga,
            home_fifa,
            away_fifa,
            home_fifa - away_fifa,
            home_advantage
        ]
        
        return np.array(features).reshape(1, -1)
    
    def prepare_training_data(self, historical_matches):
        """Prepare training data from historical matches"""
        
        X = []
        y = []
        
        team_list = list(set(
            list(historical_matches['home_team'].unique()) + 
            list(historical_matches['away_team'].unique())
        ))
        
        for _, match in historical_matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            features = self.create_features(home_team, away_team, historical_matches)
            X.append(features.flatten())
            
            # Target: 0=away win, 1=draw, 2=home win
            if home_goals > away_goals:
                target = 2
            elif home_goals < away_goals:
                target = 0
            else:
                target = 1
            
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self):
        """Train the model"""
        
        print("\n" + "="*70)
        print("TRAINING MATCH OUTCOME PREDICTION MODEL")
        print("="*70 + "\n")
        
        # Prepare training data
        X, y = self.prepare_training_data(self.data['historical_matches'])
        
        if len(X) == 0:
            print("⚠️  Insufficient training data. Using mock data.")
            X = np.random.rand(100, 11)
            y = np.random.randint(0, 3, 100)
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Target classes: {len(np.unique(y))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and use ensemble
        print("\nTraining models...")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_score = accuracy_score(y_test, rf_pred)
        print(f"  Random Forest Accuracy: {rf_score:.3f}")
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_score = accuracy_score(y_test, gb_pred)
        print(f"  Gradient Boosting Accuracy: {gb_score:.3f}")
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_score = accuracy_score(y_test, lr_pred)
        print(f"  Logistic Regression Accuracy: {lr_score:.3f}")
        
        # Use best model
        if rf_score >= gb_score and rf_score >= lr_score:
            self.model = rf_model
            best_score = rf_score
            best_model_name = "Random Forest"
        elif gb_score >= lr_score:
            self.model = gb_model
            best_score = gb_score
            best_model_name = "Gradient Boosting"
        else:
            self.model = lr_model
            best_score = lr_score
            best_model_name = "Logistic Regression"
        
        print(f"\n✓ Best Model: {best_model_name} (Accuracy: {best_score:.3f})")
        
        # Calculate additional metrics
        y_pred = self.model.predict(X_test_scaled)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        # Save model
        self.save_model()
        
        return self.model, best_score
    
    def predict_match(self, home_team, away_team):
        """Predict a single match outcome"""
        
        if self.model is None:
            self.train()
        
        features = self.create_features(home_team, away_team, self.data['historical_matches'])
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
        else:
            # For models without predict_proba
            pred = self.model.predict(features_scaled)[0]
            probabilities = np.zeros(3)
            probabilities[pred] = 1.0
        
        outcome = self.model.predict(features_scaled)[0]
        
        outcomes = {
            0: f"{away_team} Win",
            1: "Draw",
            2: f"{home_team} Win"
        }
        
        return outcomes[outcome], probabilities
    
    def save_model(self):
        """Save trained model"""
        
        os.makedirs('models', exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"\n✓ Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model"""
        
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
            print(f"✓ Model loaded from {self.model_path}")
            return True
        return False


class TournamentSimulator:
    """Simulate entire tournament using trained model"""
    
    def __init__(self):
        self.model = MatchOutcomeModel()
        self.results = {}
    
    def simulate_group_stage(self, groups):
        """Simulate group stage matches"""
        
        print("\n" + "="*70)
        print("SIMULATING GROUP STAGE")
        print("="*70 + "\n")
        
        group_standings = {}
        
        for group, teams in groups.items():
            print(f"\nGroup {group}:")
            print("-" * 50)
            
            # Initialize standings
            standings = {team: {'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0} for team in teams}
            
            # Play round-robin
            for i, team1 in enumerate(teams):
                for team2 in teams[i+1:]:
                    outcome, probs = self.model.predict_match(team1, team2)
                    
                    # Simulate goals
                    goals1 = np.random.poisson(1.5 * probs[2])
                    goals2 = np.random.poisson(1.5 * probs[0])
                    
                    print(f"{team1:20} {goals1}-{goals2} {team2:20} ({outcome})")
                    
                    # Update standings
                    standings[team1]['GF'] += goals1
                    standings[team1]['GA'] += goals2
                    standings[team2]['GF'] += goals2
                    standings[team2]['GA'] += goals1
                    
                    if goals1 > goals2:
                        standings[team1]['W'] += 1
                        standings[team2]['L'] += 1
                        standings[team1]['Pts'] += 3
                    elif goals1 < goals2:
                        standings[team2]['W'] += 1
                        standings[team1]['L'] += 1
                        standings[team2]['Pts'] += 3
                    else:
                        standings[team1]['D'] += 1
                        standings[team2]['D'] += 1
                        standings[team1]['Pts'] += 1
                        standings[team2]['Pts'] += 1
            
            # Sort teams
            sorted_teams = sorted(standings.items(), key=lambda x: (-x[1]['Pts'], -(x[1]['GF'] - x[1]['GA'])))
            group_standings[group] = sorted_teams
            
            # Display standings
            print(f"\nGroup {group} Final Standings:")
            for rank, (team, stats) in enumerate(sorted_teams, 1):
                gd = stats['GF'] - stats['GA']
                print(f"  {rank}. {team:15} {stats['Pts']}pts ({stats['W']}-{stats['D']}-{stats['L']}) GF:{stats['GF']} GA:{stats['GA']} GD:{gd:+d}")
        
        return group_standings
    
    def simulate_full_tournament(self, groups):
        """Simulate entire tournament"""
        
        print("\n" + "#"*70)
        print("TOURNAMENT SIMULATION".center(70))
        print("#"*70)
        
        group_standings = self.simulate_group_stage(groups)
        
        # Get qualified teams
        qualified = []
        for group, standings in group_standings.items():
            qualified.append(standings[0][0])
            qualified.append(standings[1][0])
        
        print("\n" + "="*70)
        print("QUALIFIED TEAMS FOR KNOCKOUT STAGE")
        print("="*70)
        for i, team in enumerate(qualified, 1):
            print(f"{i:2}. {team}")
        
        return qualified


if __name__ == "__main__":
    model = MatchOutcomeModel()
    accuracy = model.train()
    
    # Test prediction
    print("\n" + "-"*70)
    print("TEST PREDICTIONS")
    print("-"*70)
    test_matches = [
        ("Argentina", "Mexico"),
        ("France", "England"),
        ("Brazil", "Germany")
    ]
    
    for home, away in test_matches:
        outcome, probs = model.predict_match(home, away)
        print(f"{home} vs {away}: {outcome}")
        print(f"  Probabilities: Home {probs[2]:.1%}, Draw {probs[1]:.1%}, Away {probs[0]:.1%}")

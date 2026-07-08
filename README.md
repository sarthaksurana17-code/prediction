# World Cup 2026 Prediction Model

A comprehensive machine learning-based prediction model for the FIFA World Cup 2026, featuring:

## 🎯 Features

- **Tournament Predictions**: Predicting match outcomes and advancing teams through all stages
- **Elo-Based Team Strength**: Dynamic team ratings that update after every match, capturing current form rather than static historical averages
- **Top Goalscorer Prediction**: Identifying the most likely top scorer with goal projections
- **Golden Ball Winner Prediction**: Predicting the best player of the tournament
- **Machine Learning Models**: Ensemble methods for robust predictions
- **Backtested Accuracy**: Walk-forward validation against 2014, 2018, and 2022 World Cups
- **Interactive Dashboard**: Live Streamlit app for exploring predictions and team rankings
- **Historical Analysis**: Analyzed data from past FIFA World Cups (2014, 2018, 2022)

## 📊 Current Status

- **Date**: July 8, 2026
- **Tournament Stage**: Round of 16 complete; quarterfinals set
- **Quarterfinal Schedule**:
  - France vs. Morocco — July 9
  - Spain vs. Belgium — July 10
  - Norway vs. England — July 11
  - Argentina vs. Switzerland — July 11
- **Notable**: All three co-hosts (USA, Canada, Mexico) were eliminated in the Round of 16, won by Belgium, Morocco, and England respectively. Defending champions Argentina advanced past Egypt to reach the quarterfinals.

## 🚀 Quick Start

### Prerequisites

```
pip install -r requirements.txt
```

### Run All Predictions

```
python main.py
```

### Run the Interactive Dashboard

```
streamlit run dashboard_app.py
```

### Run the Backtest (validate model accuracy against past World Cups)

```
python backtest.py
```

### Run Individual Predictions

```python
# Tournament Predictions
from src.tournament_predictor import TournamentPredictor
tournament = TournamentPredictor()
champion = tournament.run_full_tournament_simulation()

# Elo-based team strength / match win probability
from src.elo_ratings import EloRatingSystem
elo = EloRatingSystem()
elo.process_match_history(historical_matches_df)
p_win, p_draw, p_loss = elo.win_probability("Argentina", "Switzerland")

# Top Goalscorer
from src.goalscorer_predictor import GoalscorerPredictor
goalscorer = GoalscorerPredictor()
top_scorers = goalscorer.predict_top_scorers(top_n=5)

# Golden Ball
from src.golden_ball_predictor import GoldenBallPredictor
golden_ball = GoldenBallPredictor()
winner = golden_ball.predict_winner()
```

## 📁 Project Structure

```
prediction/
├── main.py                              # Entry point for all predictions
├── backtest.py                          # Walk-forward accuracy validation (2014/2018/2022)
├── dashboard_app.py                     # Streamlit interactive dashboard
├── requirements.txt                     # Project dependencies
├── README.md                            # This file
│
├── src/
│   ├── data_loader.py                  # Data loading and preprocessing
│   ├── utils.py                        # Utility functions and helpers
│   ├── model.py                        # ML model training and prediction
│   ├── elo_ratings.py                  # Elo rating system for team strength
│   ├── tournament_predictor.py         # Tournament stage predictions
│   ├── goalscorer_predictor.py         # Top goalscorer predictions
│   └── golden_ball_predictor.py        # Golden Ball predictions
│
├── notebooks/
│   └── analysis.ipynb                  # Comprehensive analysis notebook
│
├── data/
│   ├── historical_matches.csv          # Past World Cup matches (loaded programmatically)
│   ├── player_stats.csv                # Player statistics (loaded programmatically)
│   ├── team_stats.csv                  # Team statistics (loaded programmatically)
│   └── wc_2026_matches.csv             # WC 2026 match schedule
│
└── models/
    └── match_outcome_model.pkl         # Trained ML model (generated after training)
```

## 🏆 Key Predictions

### Tournament Champion

Predicted based on:

- Elo ratings (dynamic, form-adjusted team strength) and FIFA rankings
- Historical performance
- Squad depth and form
- Group/knockout stage advancement probability

### Top Goalscorer

Predicted based on:

- Player's historical scoring rate
- Team strength (affects tournament advancement)
- Player rating and form
- Expected games played
- Age and experience factors

### Golden Ball (Best Player)

Predicted based on:

- Overall player rating
- Consistent performance record
- Leadership and clutch performance
- Team advancement
- Recent form and momentum

## 🔧 How It Works

### 1. Data Processing

- Loads historical World Cup match data (2014, 2018, 2022)
- Analyzes player statistics and performance metrics
- Calculates team strength indexes using both historical averages and Elo ratings

### 2. Model Training

- Uses multiple ML algorithms:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
- Elo ratings provide an additional, form-sensitive strength feature
- Selects best performing model
- Evaluates using accuracy, precision, recall, F1 score, and Brier score

### 3. Predictions

- **Match Probability**: Calculates win/draw/loss probabilities for each match
- **Tournament Simulation**: Runs group stage and knockout stage predictions
- **Player Projections**: Estimates goals and performance for individual players

### 4. Validation

- **Backtesting**: `backtest.py` trains on all matches before a given World Cup and evaluates predictions against that tournament's real results (2014, 2018, 2022), avoiding lookahead bias from shuffled train/test splits
- Compared against a naive "always predict home win" baseline

### 5. Visualizations

- Player rating distributions
- Top scorers comparison
- Team rankings analysis (Elo leaderboard)
- Historical goals distribution
- Interactive match predictor via `dashboard_app.py`

## 📈 Model Performance

- **Accuracy**: See `backtest.py` output for current walk-forward accuracy across 2014/2018/2022 World Cups
- **Brier Score**: Reported alongside accuracy to assess probability calibration, not just outcome correctness
- **Precision/Recall**: Evaluated on test set (20% of historical data)
- **Confidence**: Based on model probability outputs

*Run `python backtest.py` and paste the printed summary here once available, e.g.: "Achieved X% outcome accuracy vs. Y% naive baseline across 3 World Cups (2014–2022), Brier score Z."*

## 🎓 Learning Resources

This project demonstrates:

- Machine Learning classification (Random Forest, Gradient Boosting)
- Elo rating systems and probabilistic forecasting
- Data preprocessing and feature engineering
- Model validation and backtesting methodology
- Python pandas, scikit-learn, and Streamlit
- Git and GitHub best practices
- Project organization and structure

## 💡 Usage Examples

### Example 1: Predict a Specific Match

```python
from src.model import MatchOutcomeModel

model = MatchOutcomeModel()
model.train()

# Predict match outcome
outcome, probabilities = model.predict_match("Argentina", "Switzerland")
print(f"Prediction: {outcome}")
print(f"Probabilities: Home {probabilities[2]:.1%}, Draw {probabilities[1]:.1%}, Away {probabilities[0]:.1%}")
```

### Example 2: Get Elo-Based Win Probability

```python
from src.elo_ratings import EloRatingSystem
from src.data_loader import DataLoader

loader = DataLoader()
matches = loader.get_all_data()["historical_matches"]

elo = EloRatingSystem()
elo.process_match_history(matches)

p_win, p_draw, p_loss = elo.win_probability("France", "Morocco")
print(f"France win: {p_win:.1%}, Draw: {p_draw:.1%}, Morocco win: {p_loss:.1%}")
```

### Example 3: Analyze Player for Golden Ball

```python
from src.golden_ball_predictor import GoldenBallPredictor

predictor = GoldenBallPredictor()
analysis = predictor.analyze_player_for_award("Kylian Mbappé")
```

### Example 4: Run Full Tournament Simulation

```python
from src.model import TournamentSimulator
from src.data_loader import DataLoader

simulator = TournamentSimulator()
loader = DataLoader()
data = loader.get_all_data()

qualified_teams = simulator.simulate_full_tournament(loader.groups)
```

## 🔄 Updating Predictions

As the tournament progresses, you can:

1. Add match results to the historical data
2. Retrain the model with new data
3. Update player statistics with tournament performance
4. Recalibrate predictions based on actual results (Elo ratings update automatically as new results are fed in)

## 📝 Notes for First-Time GitHub Users

This project follows GitHub best practices:

- Clear repository structure
- Comprehensive README
- Sample data for easy setup
- Modular code organization
- Git commits for tracking changes

### Getting Started with GitHub:

1. Clone the repo: `git clone https://github.com/sarthaksurana17-code/prediction`
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make changes and test
4. Commit changes: `git commit -m "Add your message"`
5. Push to GitHub: `git push origin feature/your-feature`
6. Create a Pull Request

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report issues
- Suggest improvements
- Add more features
- Improve documentation
- Share predictions

## 📜 License

MIT License - Feel free to use this for educational purposes

## 🙏 Acknowledgments

- Historical data from past FIFA World Cups
- Inspired by sports analytics and machine learning
- Built as a learning project for GitHub and ML

---

**Last Updated**: July 8, 2026
**Project Status**: Active Development
**Contact**: <sarthaksurana17@gmail.com>

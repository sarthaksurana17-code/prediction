# World Cup 2026 Prediction Model

A comprehensive machine learning-based prediction model for the FIFA World Cup 2026, featuring:

## 🎯 Features

- **Tournament Predictions**: Predicting match outcomes and advancing teams through all stages
- **Top Goalscorer Prediction**: Identifying the most likely top scorer with goal projections
- **Golden Ball Winner Prediction**: Predicting the best player of the tournament
- **Machine Learning Models**: Ensemble methods for robust predictions
- **Historical Analysis**: Analyzed data from past FIFA World Cups (2014, 2018, 2022)

## 📊 Current Status

- **Date**: June 19, 2026
- **Progress**: Mexico has qualified to the Round of 32
- **Tournament Stage**: Group stage in progress

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run All Predictions

```bash
python main.py
```

### Run Individual Predictions

```python
# Tournament Predictions
from src.tournament_predictor import TournamentPredictor
tournament = TournamentPredictor()
champion = tournament.run_full_tournament_simulation()

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
├── requirements.txt                     # Project dependencies
├── README.md                            # This file
│
├── src/
│   ├── data_loader.py                  # Data loading and preprocessing
│   ├── utils.py                        # Utility functions and helpers
│   ├── model.py                        # ML model training and prediction
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
- FIFA rankings and team strength
- Historical performance
- Squad depth and form
- Group stage advancement probability

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
- Calculates team strength indexes based on historical results

### 2. Model Training
- Uses multiple ML algorithms:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
- Selects best performing model
- Evaluates using accuracy, precision, recall, and F1 score

### 3. Predictions
- **Match Probability**: Calculates win/draw/loss probabilities for each match
- **Tournament Simulation**: Runs group stage and knockout stage predictions
- **Player Projections**: Estimates goals and performance for individual players

### 4. Visualizations
- Player rating distributions
- Top scorers comparison
- Team rankings analysis
- Historical goals distribution

## 📈 Model Performance

- **Accuracy**: Varies based on training data volume
- **Precision/Recall**: Evaluated on test set (20% of historical data)
- **Confidence**: Based on model probability outputs

## 🎓 Learning Resources

This project demonstrates:
- Machine Learning classification (Random Forest, Gradient Boosting)
- Data preprocessing and feature engineering
- Python pandas and scikit-learn
- Git and GitHub best practices
- Project organization and structure

## 💡 Usage Examples

### Example 1: Predict a Specific Match
```python
from src.model import MatchOutcomeModel

model = MatchOutcomeModel()
model.train()

# Predict match outcome
outcome, probabilities = model.predict_match("Argentina", "Mexico")
print(f"Prediction: {outcome}")
print(f"Probabilities: Home {probabilities[2]:.1%}, Draw {probabilities[1]:.1%}, Away {probabilities[0]:.1%}")
```

### Example 2: Analyze Player for Golden Ball
```python
from src.golden_ball_predictor import GoldenBallPredictor

predictor = GoldenBallPredictor()
analysis = predictor.analyze_player_for_award("Kylian Mbappé")
```

### Example 3: Run Full Tournament Simulation
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
4. Recalibrate predictions based on actual results

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

**Last Updated**: June 19, 2026  
**Project Status**: Active Development  
**Contact**: sarthaksurana17@gmail.com

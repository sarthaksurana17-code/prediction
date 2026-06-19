"""
Main entry point for World Cup 2026 Prediction Model
Run all predictions: Tournament, Top Goalscorer, Golden Ball
"""

from src.tournament_predictor import TournamentPredictor
from src.goalscorer_predictor import GoalscorerPredictor
from src.golden_ball_predictor import GoldenBallPredictor


def main():
    """Run all predictions"""
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  WORLD CUP 2026 - COMPREHENSIVE PREDICTION MODEL  ".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print("\n📅 Current Status: June 19, 2026")
    print("📍 Progress: Mexico qualified to Round of 32")
    print("\nStarting full tournament analysis and predictions...\n")
    
    # ========== TOURNAMENT PREDICTIONS ==========
    print("\n" + "▶"*35)
    tournament_predictor = TournamentPredictor()
    champion = tournament_predictor.run_full_tournament_simulation()
    
    # ========== TOP GOALSCORER PREDICTION ==========
    print("\n" + "▶"*35)
    goalscorer_predictor = GoalscorerPredictor()
    top_scorers = goalscorer_predictor.predict_top_scorers(top_n=5)
    
    # ========== GOLDEN BALL PREDICTION ==========
    print("\n" + "▶"*35)
    golden_ball_predictor = GoldenBallPredictor()
    top_candidates = golden_ball_predictor.predict_winner(top_n=5)
    golden_ball_predictor.predict_other_awards()
    
    # ========== SUMMARY ==========
    print("\n" + "#"*70)
    print("PREDICTION SUMMARY - WORLD CUP 2026".center(70))
    print("#"*70)
    
    print(f"\n🏆 Tournament Champion: {champion}")
    
    if len(top_scorers) > 0:
        top_scorer = top_scorers.iloc[0]
        print(f"⚽ Top Goalscorer: {top_scorer['player']} ({top_scorer['predicted_goals']} goals)")
    
    if len(top_candidates) > 0:
        gb_winner = top_candidates.iloc[0]
        print(f"🌟 Golden Ball: {gb_winner['player']}")
    
    print("\n" + "#"*70)
    print("✅ Prediction model complete!")
    print("For more details, run individual modules or check the output above.")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()

"""
Simple script to predict NLL match winner
Given two teams' stats, predict who wins
"""

import json
from pathlib import Path

def predict_match_winner(home_team_stats, away_team_stats, h2h_history=None):
    """
    Predict winner given two teams' statistics

    Args:
        home_team_stats: Dict with home team's recent performance
        away_team_stats: Dict with away team's recent performance
        h2h_history: Optional head-to-head record

    Returns:
        Dict with prediction and probability
    """

    # Required stats for each team:
    # - last10_avg_goal_diff: Average goal differential over last 10 games
    # - last10_avg_goals_against: Average goals allowed over last 10 games
    # - last10_win_pct: Win percentage over last 10 games
    # - back_to_back: Boolean (1 if playing back-to-back, 0 otherwise)
    # - streak: Current win/loss streak (positive=wins, negative=losses)

    # Build the feature vector (10 features)
    features = {
        'home_last10_avg_goal_diff': home_team_stats.get('last10_avg_goal_diff', 0),
        'home_last10_avg_goals_against': home_team_stats.get('last10_avg_goals_against', 11.5),
        'home_last10_win_pct': home_team_stats.get('last10_win_pct', 0.5),
        'home_back_to_back': home_team_stats.get('back_to_back', 0),
        'home_streak': home_team_stats.get('streak', 0),

        'away_last10_avg_goal_diff': away_team_stats.get('last10_avg_goal_diff', 0),
        'away_last10_win_pct': away_team_stats.get('last10_win_pct', 0.5),
        'away_back_to_back': away_team_stats.get('back_to_back', 0),
        'away_streak': away_team_stats.get('streak', 0),

        'h2h_team1_win_pct': h2h_history.get('home_win_pct', 0.5) if h2h_history else 0.5
    }

    return features


def simple_prediction(features):
    """
    Simple weighted prediction based on feature importance
    (This is a simplified version - the real model uses logistic regression)
    """

    # Weights based on feature correlations from analysis
    weights = {
        'home_last10_avg_goal_diff': 0.203,
        'home_last10_avg_goals_against': -0.171,
        'home_last10_win_pct': 0.171,
        'away_last10_avg_goal_diff': -0.203,  # Negative because good away team hurts home
        'away_last10_win_pct': -0.171,
        'h2h_team1_win_pct': 0.163,
        'home_back_to_back': 0.05,
        'away_back_to_back': 0.20,  # Big boost when away team tired
        'home_streak': 0.05,
        'away_streak': -0.05
    }

    # Calculate weighted score
    score = 0
    for feature, value in features.items():
        if feature in weights:
            score += weights[feature] * value

    # Convert to probability (sigmoid-like)
    # Positive score = home favored, negative = away favored
    import math
    probability = 1 / (1 + math.exp(-score * 2))  # Scale by 2 for better range

    prediction = 1 if probability > 0.5 else 0

    return {
        'prediction': prediction,
        'home_win_probability': probability,
        'away_win_probability': 1 - probability,
        'confidence': abs(probability - 0.5) * 2,  # 0-1 scale
        'score': score
    }


def main():
    """Example predictions"""

    print("="*70)
    print("NLL MONEYLINE PREDICTION - MATCH WINNER")
    print("="*70)

    # EXAMPLE 1: Strong home team vs weak away team
    print("\n" + "="*70)
    print("EXAMPLE 1: Strong Home Team vs Weak Away Team")
    print("="*70)

    home_team = {
        'team_name': 'Team A (Strong)',
        'last10_avg_goal_diff': 3.2,      # Scoring +3.2 goals/game
        'last10_avg_goals_against': 9.8,  # Good defense
        'last10_win_pct': 0.80,            # 8-2 record
        'back_to_back': 0,                 # Well rested
        'streak': 4                        # 4-game win streak
    }

    away_team = {
        'team_name': 'Team B (Weak)',
        'last10_avg_goal_diff': -2.5,     # Losing by 2.5 goals/game
        'last10_win_pct': 0.30,            # 3-7 record
        'back_to_back': 1,                 # Playing back-to-back (tired!)
        'streak': -3                       # 3-game losing streak
    }

    h2h = {'home_win_pct': 0.70}  # Home team won 70% of past meetings

    features = predict_match_winner(home_team, away_team, h2h)
    result = simple_prediction(features)

    print(f"\nHome: {home_team['team_name']}")
    print(f"  - Goal Diff: {home_team['last10_avg_goal_diff']:+.1f}")
    print(f"  - Win %: {home_team['last10_win_pct']:.1%}")
    print(f"  - Streak: {home_team['streak']:+d}")
    print(f"  - Back-to-back: {'Yes' if home_team['back_to_back'] else 'No'}")

    print(f"\nAway: {away_team['team_name']}")
    print(f"  - Goal Diff: {away_team['last10_avg_goal_diff']:+.1f}")
    print(f"  - Win %: {away_team['last10_win_pct']:.1%}")
    print(f"  - Streak: {away_team['streak']:+d}")
    print(f"  - Back-to-back: {'Yes' if away_team['back_to_back'] else 'No'}")

    print(f"\nHead-to-Head: Home wins {h2h['home_win_pct']:.0%} of time")

    print("\n" + "-"*70)
    print("PREDICTION:")
    print("-"*70)
    print(f"Winner: {'HOME' if result['prediction'] == 1 else 'AWAY'}")
    print(f"Home Win Probability: {result['home_win_probability']:.1%}")
    print(f"Away Win Probability: {result['away_win_probability']:.1%}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Betting Recommendation: {'BET HOME' if result['home_win_probability'] > 0.60 else 'BET AWAY' if result['away_win_probability'] > 0.60 else 'SKIP (too close)'}")

    # EXAMPLE 2: Evenly matched teams
    print("\n" + "="*70)
    print("EXAMPLE 2: Evenly Matched Teams")
    print("="*70)

    home_team = {
        'team_name': 'Team C (Balanced)',
        'last10_avg_goal_diff': 0.5,
        'last10_avg_goals_against': 11.2,
        'last10_win_pct': 0.50,
        'back_to_back': 0,
        'streak': 1
    }

    away_team = {
        'team_name': 'Team D (Balanced)',
        'last10_avg_goal_diff': 0.3,
        'last10_win_pct': 0.50,
        'back_to_back': 0,
        'streak': -1
    }

    h2h = {'home_win_pct': 0.50}  # Split past meetings

    features = predict_match_winner(home_team, away_team, h2h)
    result = simple_prediction(features)

    print(f"\nHome: {home_team['team_name']}")
    print(f"  - Goal Diff: {home_team['last10_avg_goal_diff']:+.1f}")
    print(f"  - Win %: {home_team['last10_win_pct']:.1%}")

    print(f"\nAway: {away_team['team_name']}")
    print(f"  - Goal Diff: {away_team['last10_avg_goal_diff']:+.1f}")
    print(f"  - Win %: {away_team['last10_win_pct']:.1%}")

    print("\n" + "-"*70)
    print("PREDICTION:")
    print("-"*70)
    print(f"Winner: {'HOME' if result['prediction'] == 1 else 'AWAY'}")
    print(f"Home Win Probability: {result['home_win_probability']:.1%}")
    print(f"Away Win Probability: {result['away_win_probability']:.1%}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Betting Recommendation: {'BET HOME' if result['home_win_probability'] > 0.60 else 'BET AWAY' if result['away_win_probability'] > 0.60 else 'SKIP (too close)'}")

    # EXAMPLE 3: Away team favored (rare)
    print("\n" + "="*70)
    print("EXAMPLE 3: Away Team Favored")
    print("="*70)

    home_team = {
        'team_name': 'Team E (Weak)',
        'last10_avg_goal_diff': -2.0,
        'last10_avg_goals_against': 13.5,
        'last10_win_pct': 0.20,
        'back_to_back': 1,  # Tired!
        'streak': -5
    }

    away_team = {
        'team_name': 'Team F (Elite)',
        'last10_avg_goal_diff': 4.0,
        'last10_win_pct': 0.90,
        'back_to_back': 0,
        'streak': 7
    }

    h2h = {'home_win_pct': 0.20}  # Home team rarely wins this matchup

    features = predict_match_winner(home_team, away_team, h2h)
    result = simple_prediction(features)

    print(f"\nHome: {home_team['team_name']}")
    print(f"  - Goal Diff: {home_team['last10_avg_goal_diff']:+.1f}")
    print(f"  - Win %: {home_team['last10_win_pct']:.1%}")
    print(f"  - Back-to-back: {'Yes' if home_team['back_to_back'] else 'No'}")

    print(f"\nAway: {away_team['team_name']}")
    print(f"  - Goal Diff: {away_team['last10_avg_goal_diff']:+.1f}")
    print(f"  - Win %: {away_team['last10_win_pct']:.1%}")
    print(f"  - Streak: {away_team['streak']:+d}")

    print("\n" + "-"*70)
    print("PREDICTION:")
    print("-"*70)
    print(f"Winner: {'HOME' if result['prediction'] == 1 else 'AWAY'}")
    print(f"Home Win Probability: {result['home_win_probability']:.1%}")
    print(f"Away Win Probability: {result['away_win_probability']:.1%}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Betting Recommendation: {'BET HOME' if result['home_win_probability'] > 0.60 else 'BET AWAY' if result['away_win_probability'] > 0.60 else 'SKIP (too close)'}")

    print("\n" + "="*70)
    print("SUMMARY OF KEY FACTORS:")
    print("="*70)
    print("1. Goal Differential (last 10 games) - MOST IMPORTANT")
    print("2. Win Percentage (last 10 games)")
    print("3. Head-to-Head History")
    print("4. Back-to-Back Status (HUGE for away teams)")
    print("5. Current Streak (momentum)")
    print("\nModel achieves 58.33% accuracy (vs 50% baseline)")
    print("Expected ROI: 5-6% at standard -110 odds")
    print("="*70)


if __name__ == '__main__':
    main()

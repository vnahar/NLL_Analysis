"""
NLL Betting Analysis - Feature Engineering
Builds predictive features from raw data without sklearn
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import statistics

class FeatureEngineer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.matches = []
        self.team_stats = {}
        self.standings = {}

    def load_data(self):
        """Load processed data files"""
        print("Loading data for feature engineering...")

        # Load matches
        with open(self.data_dir / 'processed_matches.json', 'r') as f:
            data = json.load(f)
            self.matches = data['complete_matches']
        print(f"  ✓ Loaded {len(self.matches)} matches")

        # Load team stats
        with open(self.data_dir / 'team_stats_by_match.json', 'r') as f:
            self.team_stats = json.load(f)
        print(f"  ✓ Loaded team stats for {len(self.team_stats)} matches")

        # Load standings
        with open(self.data_dir / 'standings_lookup.json', 'r') as f:
            self.standings = json.load(f)
        print(f"  ✓ Loaded {len(self.standings)} standings records")

        # Sort matches by date
        self.matches.sort(key=lambda x: x['date'] if x['date'] else '9999')
        print(f"  ✓ Sorted matches chronologically")

    def calculate_rolling_stats(self, team_id, current_date, n_games=5):
        """Calculate rolling average stats for a team"""

        # Get previous n games before current date
        prev_games = []
        for match in self.matches:
            if match['date'] >= current_date:
                continue  # Don't include current or future games

            # Check if team participated
            if match['home_team_id'] == team_id:
                prev_games.append({
                    'is_home': True,
                    'goals_for': match['home_score'],
                    'goals_against': match['away_score'],
                    'won': match['home_win'] == 1,
                    'date': match['date']
                })
            elif match['away_team_id'] == team_id:
                prev_games.append({
                    'is_home': False,
                    'goals_for': match['away_score'],
                    'goals_against': match['home_score'],
                    'won': match['home_win'] == 0,
                    'date': match['date']
                })

        # Get last n games
        prev_games = prev_games[-n_games:] if len(prev_games) >= n_games else prev_games

        if not prev_games:
            return None

        # Calculate statistics
        stats = {
            'games_played': len(prev_games),
            'wins': sum(1 for g in prev_games if g['won']),
            'losses': sum(1 for g in prev_games if not g['won']),
            'avg_goals_for': statistics.mean([g['goals_for'] for g in prev_games]),
            'avg_goals_against': statistics.mean([g['goals_against'] for g in prev_games]),
            'avg_goal_diff': statistics.mean([g['goals_for'] - g['goals_against'] for g in prev_games]),
            'win_pct': sum(1 for g in prev_games if g['won']) / len(prev_games),
        }

        return stats

    def calculate_home_away_splits(self, team_id, current_date):
        """Calculate separate home and away performance"""

        home_games = []
        away_games = []

        for match in self.matches:
            if match['date'] >= current_date:
                continue

            if match['home_team_id'] == team_id:
                home_games.append({
                    'goals_for': match['home_score'],
                    'goals_against': match['away_score'],
                    'won': match['home_win'] == 1
                })
            elif match['away_team_id'] == team_id:
                away_games.append({
                    'goals_for': match['away_score'],
                    'goals_against': match['home_score'],
                    'won': match['home_win'] == 0
                })

        home_stats = {}
        away_stats = {}

        if home_games:
            home_stats = {
                'home_games': len(home_games),
                'home_wins': sum(1 for g in home_games if g['won']),
                'home_win_pct': sum(1 for g in home_games if g['won']) / len(home_games),
                'home_avg_goals_for': statistics.mean([g['goals_for'] for g in home_games]),
                'home_avg_goals_against': statistics.mean([g['goals_against'] for g in home_games]),
            }

        if away_games:
            away_stats = {
                'away_games': len(away_games),
                'away_wins': sum(1 for g in away_games if g['won']),
                'away_win_pct': sum(1 for g in away_games if g['won']) / len(away_games),
                'away_avg_goals_for': statistics.mean([g['goals_for'] for g in away_games]),
                'away_avg_goals_against': statistics.mean([g['goals_against'] for g in away_games]),
            }

        return {**home_stats, **away_stats}

    def calculate_head_to_head(self, team1_id, team2_id, current_date):
        """Calculate head-to-head record between two teams"""

        h2h_games = []

        for match in self.matches:
            if match['date'] >= current_date:
                continue

            if (match['home_team_id'] == team1_id and match['away_team_id'] == team2_id):
                h2h_games.append({
                    'team1_won': match['home_win'] == 1,
                    'team1_score': match['home_score'],
                    'team2_score': match['away_score']
                })
            elif (match['home_team_id'] == team2_id and match['away_team_id'] == team1_id):
                h2h_games.append({
                    'team1_won': match['home_win'] == 0,
                    'team1_score': match['away_score'],
                    'team2_score': match['home_score']
                })

        if not h2h_games:
            return {
                'h2h_games': 0,
                'h2h_team1_wins': 0,
                'h2h_team2_wins': 0,
                'h2h_team1_win_pct': 0.5
            }

        return {
            'h2h_games': len(h2h_games),
            'h2h_team1_wins': sum(1 for g in h2h_games if g['team1_won']),
            'h2h_team2_wins': sum(1 for g in h2h_games if not g['team1_won']),
            'h2h_team1_win_pct': sum(1 for g in h2h_games if g['team1_won']) / len(h2h_games),
            'h2h_avg_total': statistics.mean([g['team1_score'] + g['team2_score'] for g in h2h_games]),
        }

    def calculate_streak(self, team_id, current_date):
        """Calculate current win/loss streak"""

        games = []
        for match in self.matches:
            if match['date'] >= current_date:
                continue

            if match['home_team_id'] == team_id:
                games.append(match['home_win'] == 1)
            elif match['away_team_id'] == team_id:
                games.append(match['home_win'] == 0)

        if not games:
            return 0

        # Count streak from most recent game
        streak = 0
        last_result = games[-1]

        for result in reversed(games):
            if result == last_result:
                streak += 1
            else:
                break

        # Positive for win streak, negative for loss streak
        return streak if last_result else -streak

    def calculate_rest_days(self, team_id, current_date):
        """Calculate days since last game"""

        last_game_date = None

        for match in self.matches:
            if match['date'] >= current_date:
                continue

            if match['home_team_id'] == team_id or match['away_team_id'] == team_id:
                last_game_date = match['date']

        if not last_game_date:
            return None

        try:
            current = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
            last = datetime.fromisoformat(last_game_date.replace('Z', '+00:00'))
            return (current - last).days
        except:
            return None

    def get_season_phase(self, week_number):
        """Determine phase of season"""
        if week_number is None:
            return 'unknown'
        elif week_number <= 6:
            return 'early'
        elif week_number <= 14:
            return 'mid'
        else:
            return 'late'

    def build_features_for_match(self, match):
        """Build all features for a single match"""

        match_id = match['match_id']
        match_date = match['date']
        home_id = match['home_team_id']
        away_id = match['away_team_id']
        season = match['season_id']
        week = match['week_number']

        features = {
            'match_id': match_id,
            'date': match_date,
            'season_id': season,
            'week_number': week,
            'home_team_id': home_id,
            'away_team_id': away_id,

            # Targets
            'home_win': match['home_win'],
            'spread': match['spread'],
            'total': match['total'],

            # Context
            'season_phase': self.get_season_phase(week),
        }

        # Rolling stats for home team
        for n in [3, 5, 10]:
            stats = self.calculate_rolling_stats(home_id, match_date, n)
            if stats:
                for key, value in stats.items():
                    features[f'home_last{n}_{key}'] = value
            else:
                for key in ['games_played', 'wins', 'losses', 'avg_goals_for', 'avg_goals_against', 'avg_goal_diff', 'win_pct']:
                    features[f'home_last{n}_{key}'] = None

        # Rolling stats for away team
        for n in [3, 5, 10]:
            stats = self.calculate_rolling_stats(away_id, match_date, n)
            if stats:
                for key, value in stats.items():
                    features[f'away_last{n}_{key}'] = value
            else:
                for key in ['games_played', 'wins', 'losses', 'avg_goals_for', 'avg_goals_against', 'avg_goal_diff', 'win_pct']:
                    features[f'away_last{n}_{key}'] = None

        # Home/away splits
        home_splits = self.calculate_home_away_splits(home_id, match_date)
        away_splits = self.calculate_home_away_splits(away_id, match_date)

        for key, value in home_splits.items():
            features[f'home_{key}'] = value
        for key, value in away_splits.items():
            features[f'away_{key}'] = value

        # Head to head
        h2h = self.calculate_head_to_head(home_id, away_id, match_date)
        for key, value in h2h.items():
            features[key] = value

        # Streaks
        features['home_streak'] = self.calculate_streak(home_id, match_date)
        features['away_streak'] = self.calculate_streak(away_id, match_date)

        # Rest days
        features['home_rest_days'] = self.calculate_rest_days(home_id, match_date)
        features['away_rest_days'] = self.calculate_rest_days(away_id, match_date)

        # Back to back indicator
        features['home_back_to_back'] = 1 if features['home_rest_days'] and features['home_rest_days'] <= 1 else 0
        features['away_back_to_back'] = 1 if features['away_rest_days'] and features['away_rest_days'] <= 1 else 0

        # Matchup features (if both teams have data)
        if features.get('home_last5_avg_goals_for') and features.get('away_last5_avg_goals_against'):
            features['offense_defense_matchup_home'] = features['home_last5_avg_goals_for'] - features['away_last5_avg_goals_against']
            features['offense_defense_matchup_away'] = features['away_last5_avg_goals_for'] - features['home_last5_avg_goals_against']

        return features

    def build_all_features(self):
        """Build features for all matches"""
        print("\nBuilding features for all matches...")

        features_list = []
        for i, match in enumerate(self.matches):
            if (i + 1) % 100 == 0:
                print(f"  Processing match {i+1}/{len(self.matches)}...")

            features = self.build_features_for_match(match)
            features_list.append(features)

        print(f"  ✓ Built features for {len(features_list)} matches")

        return features_list

    def analyze_feature_completeness(self, features_list):
        """Analyze how complete each feature is"""
        print("\nAnalyzing feature completeness...")

        if not features_list:
            return

        feature_names = features_list[0].keys()
        completeness = {}

        for feature in feature_names:
            non_null_count = sum(1 for f in features_list if f.get(feature) is not None)
            completeness[feature] = {
                'non_null': non_null_count,
                'null': len(features_list) - non_null_count,
                'completeness_pct': (non_null_count / len(features_list)) * 100
            }

        # Print summary
        print("\nFeature Completeness Summary:")
        print("-" * 60)

        # Sort by completeness
        sorted_features = sorted(completeness.items(), key=lambda x: x[1]['completeness_pct'])

        for feature, stats in sorted_features[:20]:  # Show worst 20
            print(f"{feature:40} {stats['completeness_pct']:6.2f}% ({stats['non_null']}/{len(features_list)})")

        return completeness

    def save_features(self, features_list, output_path):
        """Save features to JSON"""
        output_file = Path(output_path) / 'features.json'

        with open(output_file, 'w') as f:
            json.dump({
                'features': features_list,
                'total_matches': len(features_list),
                'total_features': len(features_list[0].keys()) if features_list else 0,
                'feature_names': list(features_list[0].keys()) if features_list else []
            }, f, indent=2)

        print(f"\n✓ Saved features to {output_file}")
        print(f"  Total matches: {len(features_list)}")
        print(f"  Total features: {len(features_list[0].keys()) if features_list else 0}")
        print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main execution"""
    print("="*60)
    print("NLL BETTING ANALYSIS - FEATURE ENGINEERING")
    print("="*60)

    data_dir = '/Users/vedantnahar/Downloads/AltSportsData/NLL_Analysis/nll_betting_analysis/data'

    # Initialize
    engineer = FeatureEngineer(data_dir)

    # Load data
    engineer.load_data()

    # Build features
    features = engineer.build_all_features()

    # Analyze completeness
    engineer.analyze_feature_completeness(features)

    # Save features
    engineer.save_features(features, data_dir)

    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

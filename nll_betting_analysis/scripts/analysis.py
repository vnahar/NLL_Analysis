"""
NLL Betting Analysis - Exploratory Data Analysis
Analyze features and their relationships to targets
"""

import json
from pathlib import Path
import statistics
from collections import defaultdict, Counter

class DataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.features = []

    def load_features(self):
        """Load feature data"""
        print("Loading features...")
        with open(self.data_dir / 'features.json', 'r') as f:
            data = json.load(f)
            self.features = data['features']
        print(f"  ✓ Loaded {len(self.features)} matches with {data['total_features']} features")

        # Filter to complete data only
        self.complete_features = [
            f for f in self.features
            if f.get('home_last5_avg_goals_for') is not None
            and f.get('away_last5_avg_goals_for') is not None
        ]
        print(f"  ✓ {len(self.complete_features)} matches have complete rolling stats")

    def analyze_targets(self):
        """Analyze distribution of target variables"""
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS")
        print("="*60)

        data = self.complete_features

        # Moneyline
        home_wins = sum(1 for f in data if f['home_win'] == 1)
        away_wins = len(data) - home_wins
        print(f"\nMONEYLINE (Home Win):")
        print(f"  Home wins: {home_wins} ({home_wins/len(data)*100:.1f}%)")
        print(f"  Away wins: {away_wins} ({away_wins/len(data)*100:.1f}%)")
        print(f"  Baseline accuracy: {max(home_wins, away_wins)/len(data)*100:.1f}% (always pick {'home' if home_wins > away_wins else 'away'})")

        # Spread
        spreads = [f['spread'] for f in data]
        print(f"\nPOINT SPREAD:")
        print(f"  Mean: {statistics.mean(spreads):.2f} goals")
        print(f"  Median: {statistics.median(spreads):.2f} goals")
        print(f"  Std Dev: {statistics.stdev(spreads):.2f} goals")
        print(f"  Min: {min(spreads):.0f} goals")
        print(f"  Max: {max(spreads):.0f} goals")

        # Spread distribution
        spread_ranges = defaultdict(int)
        for s in spreads:
            if s < -5:
                spread_ranges['Away by 6+'] += 1
            elif s < -2:
                spread_ranges['Away by 3-5'] += 1
            elif s < 0:
                spread_ranges['Away by 1-2'] += 1
            elif s == 0:
                spread_ranges['Tie'] += 1
            elif s <= 2:
                spread_ranges['Home by 1-2'] += 1
            elif s <= 5:
                spread_ranges['Home by 3-5'] += 1
            else:
                spread_ranges['Home by 6+'] += 1

        print(f"\n  Spread Distribution:")
        for range_name, count in sorted(spread_ranges.items(), key=lambda x: x[1], reverse=True):
            print(f"    {range_name}: {count} ({count/len(data)*100:.1f}%)")

        # Totals
        totals = [f['total'] for f in data]
        print(f"\nTOTAL POINTS:")
        print(f"  Mean: {statistics.mean(totals):.2f} goals")
        print(f"  Median: {statistics.median(totals):.2f} goals")
        print(f"  Std Dev: {statistics.stdev(totals):.2f} goals")
        print(f"  Min: {min(totals):.0f} goals")
        print(f"  Max: {max(totals):.0f} goals")

        # Total distribution
        total_ranges = defaultdict(int)
        for t in totals:
            if t < 20:
                total_ranges['Under 20'] += 1
            elif t < 23:
                total_ranges['20-22'] += 1
            elif t < 26:
                total_ranges['23-25'] += 1
            elif t < 29:
                total_ranges['26-28'] += 1
            else:
                total_ranges['29+'] += 1

        print(f"\n  Total Distribution:")
        for range_name, count in sorted(total_ranges.items(), key=lambda x: x[1], reverse=True):
            print(f"    {range_name}: {count} ({count/len(data)*100:.1f}%)")

    def calculate_correlation(self, x_values, y_values):
        """Calculate Pearson correlation coefficient"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0

        n = len(x_values)
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(y_values)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        denominator_x = sum((x - mean_x) ** 2 for x in x_values) ** 0.5
        denominator_y = sum((y - mean_y) ** 2 for y in y_values) ** 0.5

        if denominator_x == 0 or denominator_y == 0:
            return 0

        return numerator / (denominator_x * denominator_y)

    def analyze_feature_correlations(self):
        """Analyze correlations between features and targets"""
        print("\n" + "="*60)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*60)

        data = self.complete_features

        # Get numeric features
        numeric_features = []
        sample = data[0]
        for key in sample.keys():
            if key not in ['match_id', 'date', 'season_id', 'home_team_id', 'away_team_id', 'season_phase',
                          'home_win', 'spread', 'total']:
                # Check if numeric
                if isinstance(sample[key], (int, float)) and sample[key] is not None:
                    numeric_features.append(key)

        print(f"\nAnalyzing {len(numeric_features)} numeric features...")

        # Moneyline correlations
        print("\n" + "-"*60)
        print("TOP 15 FEATURES FOR MONEYLINE (Home Win)")
        print("-"*60)

        moneyline_corrs = []
        for feature in numeric_features:
            values = []
            targets = []
            for f in data:
                if f.get(feature) is not None:
                    values.append(f[feature])
                    targets.append(f['home_win'])

            if len(values) > 10:
                corr = self.calculate_correlation(values, targets)
                moneyline_corrs.append((feature, corr, len(values)))

        moneyline_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        for i, (feature, corr, count) in enumerate(moneyline_corrs[:15], 1):
            print(f"{i:2d}. {feature:45} {corr:7.4f} (n={count})")

        # Spread correlations
        print("\n" + "-"*60)
        print("TOP 15 FEATURES FOR POINT SPREAD")
        print("-"*60)

        spread_corrs = []
        for feature in numeric_features:
            values = []
            targets = []
            for f in data:
                if f.get(feature) is not None:
                    values.append(f[feature])
                    targets.append(f['spread'])

            if len(values) > 10:
                corr = self.calculate_correlation(values, targets)
                spread_corrs.append((feature, corr, len(values)))

        spread_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        for i, (feature, corr, count) in enumerate(spread_corrs[:15], 1):
            print(f"{i:2d}. {feature:45} {corr:7.4f} (n={count})")

        # Total correlations
        print("\n" + "-"*60)
        print("TOP 15 FEATURES FOR TOTAL POINTS")
        print("-"*60)

        total_corrs = []
        for feature in numeric_features:
            values = []
            targets = []
            for f in data:
                if f.get(feature) is not None:
                    values.append(f[feature])
                    targets.append(f['total'])

            if len(values) > 10:
                corr = self.calculate_correlation(values, targets)
                total_corrs.append((feature, corr, len(values)))

        total_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        for i, (feature, corr, count) in enumerate(total_corrs[:15], 1):
            print(f"{i:2d}. {feature:45} {corr:7.4f} (n={count})")

        return {
            'moneyline': moneyline_corrs,
            'spread': spread_corrs,
            'total': total_corrs
        }

    def analyze_context_effects(self):
        """Analyze impact of context features"""
        print("\n" + "="*60)
        print("CONTEXT EFFECTS ANALYSIS")
        print("="*60)

        data = self.complete_features

        # Season phase
        print("\nSEASON PHASE:")
        phases = defaultdict(list)
        for f in data:
            phases[f['season_phase']].append(f)

        for phase in ['early', 'mid', 'late']:
            phase_data = phases[phase]
            if phase_data:
                home_wins = sum(1 for f in phase_data if f['home_win'] == 1)
                avg_total = statistics.mean([f['total'] for f in phase_data])
                avg_spread = statistics.mean([abs(f['spread']) for f in phase_data])

                print(f"  {phase.upper():6} ({len(phase_data):3} games): Home win% = {home_wins/len(phase_data)*100:5.1f}%, "
                      f"Avg total = {avg_total:5.2f}, Avg |spread| = {avg_spread:4.2f}")

        # Back-to-back games
        print("\nBACK-TO-BACK GAMES:")

        b2b_home = [f for f in data if f.get('home_back_to_back') == 1]
        rest_home = [f for f in data if f.get('home_back_to_back') == 0 and f.get('home_rest_days')]

        if b2b_home:
            home_win_b2b = sum(1 for f in b2b_home if f['home_win'] == 1) / len(b2b_home) * 100
            print(f"  Home team on B2B ({len(b2b_home):3} games): {home_win_b2b:5.1f}% win rate")

        if rest_home:
            home_win_rest = sum(1 for f in rest_home if f['home_win'] == 1) / len(rest_home) * 100
            print(f"  Home team with rest ({len(rest_home):3} games): {home_win_rest:5.1f}% win rate")

        b2b_away = [f for f in data if f.get('away_back_to_back') == 1]
        rest_away = [f for f in data if f.get('away_back_to_back') == 0 and f.get('away_rest_days')]

        if b2b_away:
            away_win_b2b = sum(1 for f in b2b_away if f['home_win'] == 0) / len(b2b_away) * 100
            print(f"  Away team on B2B ({len(b2b_away):3} games): {away_win_b2b:5.1f}% win rate")

        if rest_away:
            away_win_rest = sum(1 for f in rest_away if f['home_win'] == 0) / len(rest_away) * 100
            print(f"  Away team with rest ({len(rest_away):3} games): {away_win_rest:5.1f}% win rate")

        # Streaks
        print("\nWIN STREAK EFFECTS:")

        streak_buckets = {
            'Hot (3+ wins)': [],
            'Warm (1-2 wins)': [],
            'Cold (-1 to -2)': [],
            'Ice Cold (-3+)': []
        }

        for f in data:
            home_streak = f.get('home_streak', 0)
            if home_streak >= 3:
                streak_buckets['Hot (3+ wins)'].append(('home', f))
            elif home_streak >= 1:
                streak_buckets['Warm (1-2 wins)'].append(('home', f))
            elif home_streak >= -2:
                streak_buckets['Cold (-1 to -2)'].append(('home', f))
            else:
                streak_buckets['Ice Cold (-3+)'].append(('home', f))

            away_streak = f.get('away_streak', 0)
            if away_streak >= 3:
                streak_buckets['Hot (3+ wins)'].append(('away', f))
            elif away_streak >= 1:
                streak_buckets['Warm (1-2 wins)'].append(('away', f))
            elif away_streak >= -2:
                streak_buckets['Cold (-1 to -2)'].append(('away', f))
            else:
                streak_buckets['Ice Cold (-3+)'].append(('away', f))

        for bucket_name, bucket_data in streak_buckets.items():
            if bucket_data:
                wins = sum(1 for team, f in bucket_data
                          if (team == 'home' and f['home_win'] == 1) or (team == 'away' and f['home_win'] == 0))
                print(f"  {bucket_name:20} ({len(bucket_data):3} instances): {wins/len(bucket_data)*100:5.1f}% win rate")

    def analyze_team_performance(self):
        """Analyze individual team performance"""
        print("\n" + "="*60)
        print("TEAM PERFORMANCE ANALYSIS")
        print("="*60)

        data = self.complete_features

        team_stats = defaultdict(lambda: {
            'home_games': 0, 'home_wins': 0, 'home_goals_for': 0, 'home_goals_against': 0,
            'away_games': 0, 'away_wins': 0, 'away_goals_for': 0, 'away_goals_against': 0
        })

        for f in data:
            home_id = f['home_team_id']
            away_id = f['away_team_id']

            # Home team stats
            team_stats[home_id]['home_games'] += 1
            team_stats[home_id]['home_wins'] += f['home_win']
            team_stats[home_id]['home_goals_for'] += f['total'] - f['spread']  # away score
            team_stats[home_id]['home_goals_for'] += f['spread'] + (f['total'] - f['spread'])  # home score
            team_stats[home_id]['home_goals_against'] += f['total'] - f['spread']

            # Away team stats
            team_stats[away_id]['away_games'] += 1
            team_stats[away_id]['away_wins'] += (1 - f['home_win'])
            team_stats[away_id]['away_goals_for'] += f['total'] - f['spread']
            team_stats[away_id]['away_goals_against'] += (f['total'] - (f['total'] - f['spread']))

        print(f"\nTotal teams analyzed: {len(team_stats)}")

        # Calculate overall records
        team_records = []
        for team_id, stats in team_stats.items():
            total_games = stats['home_games'] + stats['away_games']
            total_wins = stats['home_wins'] + stats['away_wins']

            if total_games > 0:
                team_records.append({
                    'team_id': team_id,
                    'games': total_games,
                    'wins': total_wins,
                    'losses': total_games - total_wins,
                    'win_pct': total_wins / total_games,
                    'home_win_pct': stats['home_wins'] / stats['home_games'] if stats['home_games'] > 0 else 0,
                    'away_win_pct': stats['away_wins'] / stats['away_games'] if stats['away_games'] > 0 else 0,
                })

        # Top teams
        team_records.sort(key=lambda x: x['win_pct'], reverse=True)

        print("\nTOP 10 TEAMS (by win %):")
        print(f"{'Team ID':10} {'W-L':10} {'Win%':7} {'Home%':7} {'Away%':7}")
        print("-" * 60)
        for team in team_records[:10]:
            print(f"{team['team_id']:10} {team['wins']:3}-{team['losses']:<6} "
                  f"{team['win_pct']*100:6.1f}% {team['home_win_pct']*100:6.1f}% {team['away_win_pct']*100:6.1f}%")

        print("\nBOTTOM 10 TEAMS (by win %):")
        print(f"{'Team ID':10} {'W-L':10} {'Win%':7} {'Home%':7} {'Away%':7}")
        print("-" * 60)
        for team in team_records[-10:]:
            print(f"{team['team_id']:10} {team['wins']:3}-{team['losses']:<6} "
                  f"{team['win_pct']*100:6.1f}% {team['home_win_pct']*100:6.1f}% {team['away_win_pct']*100:6.1f}%")

    def save_analysis_summary(self, correlations, output_path):
        """Save analysis summary to JSON"""
        output_file = Path(output_path) / 'analysis_summary.json'

        summary = {
            'total_matches': len(self.complete_features),
            'top_features': {
                'moneyline': [(f, c) for f, c, _ in correlations['moneyline'][:20]],
                'spread': [(f, c) for f, c, _ in correlations['spread'][:20]],
                'total': [(f, c) for f, c, _ in correlations['total'][:20]],
            }
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved analysis summary to {output_file}")


def main():
    """Main execution"""
    print("="*60)
    print("NLL BETTING ANALYSIS - EXPLORATORY ANALYSIS")
    print("="*60)

    data_dir = '/Users/vedantnahar/Downloads/AltSportsData/NLL_Analysis/nll_betting_analysis/data'

    analyzer = DataAnalyzer(data_dir)
    analyzer.load_features()
    analyzer.analyze_targets()
    correlations = analyzer.analyze_feature_correlations()
    analyzer.analyze_context_effects()
    analyzer.analyze_team_performance()
    analyzer.save_analysis_summary(correlations, data_dir)

    print("\n" + "="*60)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

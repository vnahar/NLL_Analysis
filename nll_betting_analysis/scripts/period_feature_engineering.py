"""
NLL Betting Analysis - Period Feature Engineering
Extracts period-level (quarter) statistics for simulation modeling
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class PeriodFeatureEngineer:
    def __init__(self, raw_data_path):
        """
        Initialize period feature engineer

        Args:
            raw_data_path: Path to raw_data.json file
        """
        self.raw_data_path = Path(raw_data_path)
        self.data = self._load_data()
        self.period_data = []
        self.match_mapping = {}

    def _load_data(self):
        """Load raw data from JSON"""
        with open(self.raw_data_path, 'r') as f:
            return json.load(f)

    def create_period_match_mapping(self):
        """
        Map period match_ids to schedule match_ids
        Period data uses different IDs than schedule data
        """
        print("Creating period-to-schedule match ID mapping...")

        schedule = self.data.get('Schedule', [])
        period_data = self.data.get('Scores Period', [])

        # Get unique period match_ids
        period_match_ids = set(p['match_id'] for p in period_data)

        # Create mapping by teams + date proximity
        mapping = {}
        unmatched = []

        for period_match_id in period_match_ids:
            # Get teams from period data for this match
            period_records = [p for p in period_data if p['match_id'] == period_match_id]
            period_teams = set(p['team_id'] for p in period_records)

            # Get match date (if available)
            # Period data doesn't have dates, so we'll match by teams only

            # Find schedule record with matching teams
            matched = False
            for sched in schedule:
                sched_teams = {sched['squads_home_id'], sched['squads_away_id']}

                # Check if teams match
                if period_teams == sched_teams:
                    # Multiple matches might have same teams, so we need better logic
                    # For now, create list of potential matches
                    if period_match_id not in mapping:
                        mapping[period_match_id] = []
                    mapping[period_match_id].append({
                        'schedule_match_id': sched['id'],
                        'date': sched.get('date_utc_match_start'),
                        'season': sched.get('season_id')
                    })
                    matched = True

            if not matched:
                unmatched.append(period_match_id)

        # For matches with multiple potential mappings, take the first one
        # (In production, would use date proximity or other logic)
        for period_id, schedule_matches in mapping.items():
            if len(schedule_matches) > 1:
                # Sort by date and take first
                schedule_matches.sort(key=lambda x: x['date'] or '')
            mapping[period_id] = schedule_matches[0] if schedule_matches else None

        print(f"  ✓ Mapped {len(mapping)} period match IDs")
        print(f"  ✗ Unmatched: {len(unmatched)} period match IDs")

        self.match_mapping = mapping
        return mapping

    def extract_period_data(self):
        """
        Extract period-level scoring data
        Returns list of period records with schedule match_ids
        """
        print("\nExtracting period-level data...")

        # Create mapping if not done yet
        if not self.match_mapping:
            self.create_period_match_mapping()

        period_raw = self.data.get('Scores Period', [])
        schedule = self.data.get('Schedule', [])

        # Create schedule lookup
        schedule_lookup = {s['id']: s for s in schedule}

        # Process each period record
        processed_periods = []

        for period in period_raw:
            period_match_id = period.get('match_id')

            # Skip if no mapping
            if period_match_id not in self.match_mapping:
                continue

            mapped = self.match_mapping[period_match_id]
            if not mapped:
                continue

            schedule_match_id = mapped['schedule_match_id']
            schedule_record = schedule_lookup.get(schedule_match_id, {})

            # Determine if home or away
            team_id = period.get('team_id')
            is_home = (team_id == schedule_record.get('squads_home_id'))

            # Get opponent ID
            if is_home:
                opponent_id = schedule_record.get('squads_away_id')
            else:
                opponent_id = schedule_record.get('squads_home_id')

            processed = {
                'match_id': schedule_match_id,
                'period_match_id': period_match_id,
                'season_id': mapped.get('season'),
                'date': mapped.get('date'),
                'team_id': team_id,
                'opponent_id': opponent_id,
                'is_home': is_home,
                'period': period.get('period', 1),
                'goals': period.get('goals', 0),
                'goals_cumulative': period.get('goals1_cumulative', 0),
                'team_type': period.get('team_type'),
            }

            processed_periods.append(processed)

        self.period_data = processed_periods

        print(f"  ✓ Extracted {len(processed_periods)} period records")
        print(f"  ✓ Covering {len(set(p['match_id'] for p in processed_periods))} unique matches")

        return processed_periods

    def calculate_team_rates(self):
        """
        Calculate offensive and defensive rates per quarter for each team

        Returns:
            Dict mapping team_id to offensive/defensive stats
        """
        print("\nCalculating team offensive/defensive rates...")

        if not self.period_data:
            self.extract_period_data()

        # Group periods by team
        team_periods = defaultdict(list)
        for period in self.period_data:
            team_periods[period['team_id']].append(period)

        team_stats = {}

        for team_id, periods in team_periods.items():
            # Filter to regular quarters only (1-4)
            regular_periods = [p for p in periods if p['period'] in [1, 2, 3, 4]]

            if not regular_periods:
                continue

            # Offensive rate: goals scored per quarter
            total_goals = sum(p['goals'] for p in regular_periods)
            offensive_rate = total_goals / len(regular_periods) if regular_periods else 0

            # Home/Away splits
            home_periods = [p for p in regular_periods if p['is_home']]
            away_periods = [p for p in regular_periods if not p['is_home']]

            home_offensive_rate = (sum(p['goals'] for p in home_periods) / len(home_periods)) if home_periods else offensive_rate
            away_offensive_rate = (sum(p['goals'] for p in away_periods) / len(away_periods)) if away_periods else offensive_rate

            # Defensive rate: need to look up opponent's goals in same matches
            defensive_goals = []
            for period in regular_periods:
                # Find opponent's period in same match
                opponent_period = next(
                    (p for p in self.period_data
                     if p['match_id'] == period['match_id']
                     and p['period'] == period['period']
                     and p['team_id'] == period['opponent_id']),
                    None
                )
                if opponent_period:
                    defensive_goals.append(opponent_period['goals'])

            defensive_rate = sum(defensive_goals) / len(defensive_goals) if defensive_goals else offensive_rate

            # Games analyzed
            unique_matches = len(set(p['match_id'] for p in regular_periods))

            team_stats[team_id] = {
                'offensive_rate_per_quarter': round(offensive_rate, 3),
                'defensive_rate_per_quarter': round(defensive_rate, 3),
                'home_offensive_rate': round(home_offensive_rate, 3),
                'away_offensive_rate': round(away_offensive_rate, 3),
                'total_quarters': len(regular_periods),
                'games_analyzed': unique_matches,
                'home_games': len(set(p['match_id'] for p in home_periods)),
                'away_games': len(set(p['match_id'] for p in away_periods))
            }

        print(f"  ✓ Calculated rates for {len(team_stats)} teams")

        return team_stats

    def build_historical_rates(self, cutoff_date=None):
        """
        Build historical offensive/defensive rates for each team
        Useful for time-series modeling

        Args:
            cutoff_date: Only use data before this date (for temporal splits)

        Returns:
            Dict mapping team_id to time-series of rates
        """
        if not self.period_data:
            self.extract_period_data()

        # Filter by cutoff date if provided
        periods = self.period_data
        if cutoff_date:
            cutoff = datetime.fromisoformat(cutoff_date.replace('Z', '+00:00'))
            periods = [
                p for p in periods
                if p['date'] and datetime.fromisoformat(p['date'].replace('Z', '+00:00')) < cutoff
            ]

        # Sort by date
        periods.sort(key=lambda x: x['date'] or '')

        # Calculate rolling rates
        team_history = defaultdict(lambda: {'offensive': [], 'defensive': [], 'dates': []})

        # Group by match to calculate rates
        match_groups = defaultdict(list)
        for p in periods:
            if p['period'] in [1, 2, 3, 4]:  # Regular quarters only
                match_groups[p['match_id']].append(p)

        # Sort matches by date
        sorted_matches = sorted(match_groups.items(),
                              key=lambda x: match_groups[x[0]][0]['date'] or '')

        # Calculate cumulative rates
        team_cumulative_goals = defaultdict(list)
        team_cumulative_allowed = defaultdict(list)

        for match_id, match_periods in sorted_matches:
            match_date = match_periods[0]['date']

            # Calculate goals for this match
            team_match_goals = defaultdict(int)
            for period in match_periods:
                team_match_goals[period['team_id']] += period['goals']

            # Update each team's history
            for team_id, goals in team_match_goals.items():
                team_cumulative_goals[team_id].append(goals / 4.0)  # Goals per quarter

                # Defensive: opponent's goals
                opponent_goals = sum(g for tid, g in team_match_goals.items() if tid != team_id)
                team_cumulative_allowed[team_id].append(opponent_goals / 4.0)

                # Store rolling average (last 10 games)
                recent_offensive = team_cumulative_goals[team_id][-10:]
                recent_defensive = team_cumulative_allowed[team_id][-10:]

                team_history[team_id]['offensive'].append(
                    sum(recent_offensive) / len(recent_offensive)
                )
                team_history[team_id]['defensive'].append(
                    sum(recent_defensive) / len(recent_defensive)
                )
                team_history[team_id]['dates'].append(match_date)

        return dict(team_history)

    def save_team_parameters(self, output_path, cutoff_date=None):
        """
        Save team parameters to JSON file

        Args:
            output_path: Path to save team_parameters.json
            cutoff_date: Only use training data before this date
        """
        print("\nGenerating team parameters...")

        # Calculate current rates
        team_stats = self.calculate_team_rates()

        # Calculate historical rates (for validation)
        team_history = self.build_historical_rates(cutoff_date)

        # Combine into output format
        output = {
            'generated_at': datetime.now().isoformat(),
            'cutoff_date': cutoff_date,
            'league_averages': self._calculate_league_averages(team_stats),
            'teams': team_stats,
            'historical_rates': team_history
        }

        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  ✓ Saved team parameters to {output_file}")
        print(f"  ✓ {len(team_stats)} teams with rate calculations")

        return output

    def _calculate_league_averages(self, team_stats):
        """Calculate league-wide average rates"""
        if not team_stats:
            return {}

        offensive_rates = [s['offensive_rate_per_quarter'] for s in team_stats.values()]
        defensive_rates = [s['defensive_rate_per_quarter'] for s in team_stats.values()]

        return {
            'avg_offensive_rate': round(sum(offensive_rates) / len(offensive_rates), 3),
            'avg_defensive_rate': round(sum(defensive_rates) / len(defensive_rates), 3),
            'avg_goals_per_quarter': round(sum(offensive_rates) / len(offensive_rates), 3),
            'total_teams': len(team_stats)
        }


def main():
    """Main execution for period feature engineering"""
    print("=" * 60)
    print("NLL Period Feature Engineering")
    print("=" * 60)

    # Paths
    raw_data_path = Path(__file__).parent.parent / 'data' / 'raw_data.json'
    output_path = Path(__file__).parent.parent / 'data' / 'team_parameters.json'

    # Initialize
    engineer = PeriodFeatureEngineer(raw_data_path)

    # Extract period data
    period_data = engineer.extract_period_data()

    # Calculate team rates
    team_stats = engineer.calculate_team_rates()

    # Save parameters (train on 2021-2024, cutoff before 2024-2025 season)
    engineer.save_team_parameters(
        output_path,
        cutoff_date='2024-11-01T00:00:00Z'  # Before 2024-25 season starts
    )

    print("\n" + "=" * 60)
    print("Period feature engineering complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
NLL Betting Analysis - Data Loader
Loads data from Excel and converts to JSON format
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class NLLDataLoader:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.data = {}

    def load_all_sheets(self):
        """Load all relevant sheets from Excel"""
        print("Loading data from Excel...")

        sheets_to_load = [
            'Schedule',
            'Scores Match',
            'Scores Period',
            'Team Stats Match Flat',
            'Team Stats Match Constructed',
            'Team Stats Season Flat',
            'Standings Flat',
            'Player Stats Match Flat',
            'Player Stats Season Flat',
            'Teams'
        ]

        for sheet in sheets_to_load:
            try:
                df = pd.read_excel(self.excel_path, sheet_name=sheet)
                print(f"  ✓ Loaded {sheet}: {len(df)} rows, {len(df.columns)} columns")
                # Convert to records and handle datetime
                records = df.to_dict('records')
                # Convert timestamps to strings
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif isinstance(value, pd.Timestamp):
                            record[key] = value.isoformat()
                        elif isinstance(value, (int, float)):
                            # Check if it's NaN for floats
                            if pd.isna(value):
                                record[key] = None
                            else:
                                record[key] = value

                self.data[sheet] = records
            except Exception as e:
                print(f"  ✗ Failed to load {sheet}: {e}")

        return self.data

    def save_to_json(self, output_path):
        """Save loaded data to JSON file"""
        output_file = Path(output_path) / 'raw_data.json'

        with open(output_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

        print(f"\n✓ Saved raw data to {output_file}")
        print(f"  Total sheets: {len(self.data)}")
        print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return str(output_file)

    def create_match_dataset(self):
        """Create unified match dataset with both teams' info"""
        print("\nCreating unified match dataset...")

        schedule = self.data.get('Schedule', [])
        scores = self.data.get('Scores Match', [])

        # Create lookup for scores by match_id (using 'id' field)
        scores_lookup = {s['id']: s for s in scores if s.get('id')}

        matches = []
        for match in schedule:
            match_id = match.get('id')
            if not match_id:
                continue

            score_data = scores_lookup.get(match_id, {})

            # Build unified match record with correct column names
            unified = {
                'match_id': match_id,
                'season_id': match.get('season_id'),
                'week_number': match.get('week_number'),
                'date': match.get('date_utc_match_start') or match.get('date_start_date'),
                'home_team_id': match.get('squads_home_id'),
                'home_team_name': match.get('squads_home_name'),
                'away_team_id': match.get('squads_away_id'),
                'away_team_name': match.get('squads_away_name'),
                'home_score': match.get('squads_home_score_score'),
                'away_score': match.get('squads_away_score_score'),
                'venue_id': match.get('venue_id'),
                'match_status': match.get('status_id'),
                'winning_team_id': match.get('winning_squad_id'),
                'home_result': 'W' if match.get('winning_squad_id') == match.get('squads_home_id') else 'L' if match.get('winning_squad_id') and match.get('squads_home_score_score') is not None else None,
                'away_result': 'W' if match.get('winning_squad_id') == match.get('squads_away_id') else 'L' if match.get('winning_squad_id') and match.get('squads_away_score_score') is not None else None,
                # Target variables
                'home_win': 1 if match.get('winning_squad_id') == match.get('squads_home_id') else 0 if match.get('winning_squad_id') and match.get('squads_home_score_score') is not None else None,
                'spread': (match.get('squads_home_score_score', 0) or 0) - (match.get('squads_away_score_score', 0) or 0) if match.get('squads_home_score_score') is not None and match.get('squads_away_score_score') is not None else None,
                'total': (match.get('squads_home_score_score', 0) or 0) + (match.get('squads_away_score_score', 0) or 0) if match.get('squads_home_score_score') is not None and match.get('squads_away_score_score') is not None else None,
            }

            matches.append(unified)

        print(f"  ✓ Created {len(matches)} unified match records")

        # Filter to matches with complete data
        complete_matches = [m for m in matches if m['home_score'] is not None and m['away_score'] is not None]
        print(f"  ✓ {len(complete_matches)} matches have complete scoring data")

        return matches, complete_matches

    def get_team_stats_by_match(self):
        """Extract team statistics per match"""
        print("\nExtracting team statistics per match...")

        team_stats = self.data.get('Team Stats Match Flat', [])

        # Group by match_id and team
        stats_by_match = {}
        for stat in team_stats:
            match_id = stat.get('match_id')
            team_id = stat.get('id')  # 'id' is the team ID in this sheet

            if not match_id or not team_id:
                continue

            if match_id not in stats_by_match:
                stats_by_match[match_id] = {}

            stats_by_match[match_id][team_id] = stat

        print(f"  ✓ Extracted stats for {len(stats_by_match)} matches")

        return stats_by_match

    def get_standings_by_date(self):
        """Extract standings information indexed by team and date"""
        print("\nExtracting standings data...")

        standings = self.data.get('Standings Flat', [])

        # Create lookup by season, week, and team
        standings_lookup = {}
        for standing in standings:
            season = standing.get('season_id')
            week = standing.get('week_number')
            team_id = standing.get('team_id')  # correct column name

            if not all([season, week, team_id]):
                continue

            key = (season, week, team_id)
            standings_lookup[key] = standing

        print(f"  ✓ Extracted {len(standings_lookup)} standing records")

        return standings_lookup

    def summarize_data(self):
        """Print summary statistics of loaded data"""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)

        for sheet_name, records in self.data.items():
            if records:
                print(f"\n{sheet_name}:")
                print(f"  Rows: {len(records)}")
                if records:
                    print(f"  Columns: {len(records[0].keys())}")

                    # Check for key identifiers
                    sample = records[0]
                    if 'match_id' in sample:
                        unique_matches = len(set(r.get('match_id') for r in records if r.get('match_id')))
                        print(f"  Unique matches: {unique_matches}")
                    if 'squad_id' in sample:
                        unique_teams = len(set(r.get('squad_id') for r in records if r.get('squad_id')))
                        print(f"  Unique teams: {unique_teams}")
                    if 'player_id' in sample:
                        unique_players = len(set(r.get('player_id') for r in records if r.get('player_id')))
                        print(f"  Unique players: {unique_players}")


def main():
    """Main execution function"""
    print("="*60)
    print("NLL BETTING ANALYSIS - DATA LOADER")
    print("="*60)

    # Paths
    excel_path = '/Users/vedantnahar/Downloads/AltSportsData/NLL_Analysis/NLL_Analytics_COMPLETE.xlsx'
    output_path = '/Users/vedantnahar/Downloads/AltSportsData/NLL_Analysis/nll_betting_analysis/data'

    # Load data
    loader = NLLDataLoader(excel_path)
    loader.load_all_sheets()

    # Save raw data
    loader.save_to_json(output_path)

    # Create match dataset
    all_matches, complete_matches = loader.create_match_dataset()

    # Save processed matches
    matches_file = Path(output_path) / 'processed_matches.json'
    with open(matches_file, 'w') as f:
        json.dump({
            'all_matches': all_matches,
            'complete_matches': complete_matches,
            'summary': {
                'total_matches': len(all_matches),
                'complete_matches': len(complete_matches),
                'missing_scores': len(all_matches) - len(complete_matches)
            }
        }, f, indent=2)
    print(f"\n✓ Saved processed matches to {matches_file}")

    # Get team stats
    team_stats = loader.get_team_stats_by_match()
    team_stats_file = Path(output_path) / 'team_stats_by_match.json'
    with open(team_stats_file, 'w') as f:
        json.dump(team_stats, f, indent=2)
    print(f"✓ Saved team stats to {team_stats_file}")

    # Get standings
    standings = loader.get_standings_by_date()
    # Convert tuple keys to strings for JSON
    standings_serializable = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in standings.items()}
    standings_file = Path(output_path) / 'standings_lookup.json'
    with open(standings_file, 'w') as f:
        json.dump(standings_serializable, f, indent=2)
    print(f"✓ Saved standings to {standings_file}")

    # Print summary
    loader.summarize_data()

    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

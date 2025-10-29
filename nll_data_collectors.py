#!/usr/bin/env python3
"""
NLL API Data Collection Functions
Modular, debuggable functions for each API endpoint type
"""

import asyncio
import json
import httpx
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configuration
BASE_URL = "https://api.nll.championdata.io/v1"
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjRYa0U1VDZRZkFpMFFaZnRRaFQweiJ9.eyJodHRwczovL3NjaGVtYS5jaGFtcGlvbmRhdGEuY29tLmF1L25hbWUiOiJKdWRlIEZlcm5hbmRlcyIsImh0dHBzOi8vc2NoZW1hLmNoYW1waW9uZGF0YS5jb20uYXUvZW1haWwiOiJqZmVybmFuZGVzQGFsdHNwb3J0c2RhdGEuY29tIiwiaXNzIjoiaHR0cHM6Ly9jaGFtcGlvbmRhdGEuYXUuYXV0aDAuY29tLyIsInN1YiI6ImF1dGgwfDY4MWE2OWMyOWY2MjU4NjkzMDBiZTM4OSIsImF1ZCI6Imh0dHBzOi8vYXBpLm5sbC5jaGFtcGlvbmRhdGEuaW8vIiwiaWF0IjoxNzYxNDM0NzgxLCJleHAiOjE3NjE0NDE5ODEsInNjb3BlIjoiIiwiYXpwIjoibjhyNlJmbmVxbFNLTXdWRUh0bU56WE5YWGlsUVE4MGYiLCJwZXJtaXNzaW9ucyI6WyJhdXRoOmJhc2ljIiwiZW52OnByb2QiLCJlbnY6cHJvZC1zYW5kYm94IiwicmVhZDpkZWZhdWx0IiwidXNhZ2U6bG93Il19.k5o8uXDagKPKABssjGgbyQQhs3xRHe_wVMn6_bZHbiSeDSnJxAhTO3nCNvmfKPImWBjdrut4tfrYmCiyCL-jSM8T6Jl_FJCzddMSXOFuwHml7mvZsf0eTxzpBS4JD8I8W7aQUKmSNONGsg5158-sFH_IQaZo3QDNdM4HWdXgAj1AazkbISkIvYqBxSc3RcDuY2Nak6lu0FfwLcaUvuiBrVEXX3axn3POYGNJqcih9mhq7-14gIjqhooJJ0HXw3s4XjI5wsHakr-vv2k_Fjsk5qmZ0-ivTJZ4Pt4oSUk5bHLdpkHFhG66j-yauXYt3llngHk7_u-e53HYYQK0MFf1Iw"

class NLLDataCollector:
    """Modular NLL data collection and processing"""

    def __init__(self):
        self.league_id = None
        self.level_id = None
        self.seasons = []
        self.match_ids = []

    async def fetch_json(self, client: httpx.AsyncClient, path: str, params: Dict[str, Any] = None) -> Any:
        """Fetch JSON from API with error handling"""
        try:
            url = f"{BASE_URL}{path}" if not path.startswith("http") else path
            headers = {"Authorization": f"Bearer {TOKEN}", "accept": "application/json"}

            response = await client.get(url, headers=headers, params=params or {})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {path}: {e}")
            return None

    async def initialize_discovery(self, client: httpx.AsyncClient) -> bool:
        """Initialize league, level, and season data"""
        print("🔍 Initializing discovery data...")

        # Get leagues
        leagues = await self.fetch_json(client, "/leagues")
        if not leagues:
            return False

        nll_league = next((lg for lg in leagues if lg.get("code") == "NLL"), leagues[0])
        self.league_id = str(nll_league["id"])
        print(f"📋 League: {nll_league.get('name')} (ID: {self.league_id})")

        # Get levels
        levels = await self.fetch_json(client, f"/leagues/{self.league_id}/levels")
        if not levels:
            return False

        if isinstance(levels, dict) and "levels" in levels:
            levels = levels["levels"]
        self.level_id = str(levels[0]["id"])
        print(f"📋 Level: {levels[0].get('name')} (ID: {self.level_id})")

        # Get seasons (2020-2024)
        seasons_data = await self.fetch_json(client, f"/leagues/{self.league_id}/levels/{self.level_id}/seasons")
        if not seasons_data:
            return False

        if isinstance(seasons_data, dict) and "seasons" in seasons_data:
            seasons = seasons_data["seasons"]
        else:
            seasons = seasons_data

        # Filter to recent seasons
        for season in seasons:
            year = season.get("startYear") or season.get("endYear")
            if year and 2020 <= int(year) <= 2024:
                self.seasons.append(season)

        print(f"📋 Found {len(self.seasons)} seasons (2020-2024)")
        return True

    async def collect_career_statistics(self, client: httpx.AsyncClient, phase: str = "REG") -> List[Dict]:
        """Collect career statistics with pagination"""
        print(f"📊 Collecting career statistics ({phase})...")

        all_players = []
        page = 1

        while True:
            data = await self.fetch_json(client, "/career/statistics/players", {
                "phase": phase,
                "page": page,
                "limit": 100
            })

            if not data or "players" not in data:
                break

            players = data["players"]
            if not players:
                break

            # Process each player's statistics
            for player in players:
                player_data = {
                    "PLAYER_ID": player.get("id"),
                    "DISPLAY_NAME": player.get("displayName"),
                    "FIRST_NAME": player.get("firstName"),
                    "LAST_NAME": player.get("lastName"),
                    "SQUAD_ID": player.get("squad", {}).get("id"),
                    "SQUAD_NAME": player.get("squad", {}).get("name"),
                    "SQUAD_CODE": player.get("squad", {}).get("code"),
                    "PHASE": phase
                }

                # Add all statistics
                for stat in player.get("statistics", []):
                    stat_code = stat.get("code", "").upper()
                    player_data[stat_code] = stat.get("value", 0)
                    player_data[f"{stat_code}_DISPLAY"] = stat.get("valueDisplay", "")

                all_players.append(player_data)

            print(f"  Page {page}: {len(players)} players")
            page += 1

            # Check if there are more pages
            if len(players) < 100:
                break

        print(f"✅ Collected {len(all_players)} career records ({phase})")
        return all_players

    async def collect_match_shots(self, client: httpx.AsyncClient, match_id: str) -> List[Dict]:
        """Collect and process match shots data"""
        data = await self.fetch_json(client, f"/matches/{match_id}/shots")
        if not data or "shots" not in data:
            return []

        processed_shots = []
        for shot in data["shots"]:
            shot_data = {
                "MATCH_ID": match_id,
                "MATCH_TRX_ID": shot.get("matchTrxId"),
                "PERIOD": shot.get("period"),
                "PERIOD_SECS": shot.get("periodSecs"),
                "PERIOD_DISPLAY": shot.get("periodDisplay"),
                "PERIOD_TIME": shot.get("periodTime"),
                "REMAINING_SECS": shot.get("remainingSecs"),
                "REMAINING_DISPLAY": shot.get("remainingDisplay"),
                "REMAINING_TIME": shot.get("remainingTime"),

                # Shot details
                "SHOT_PHASE": shot.get("details", {}).get("shotPhase"),
                "SHOT_STRENGTH": shot.get("details", {}).get("shotStrength"),
                "SHOT_VALUE": shot.get("details", {}).get("shotValue"),
                "SHOT_HAND": shot.get("details", {}).get("shotHand"),
                "SHOT_LOCATION": shot.get("details", {}).get("shotLocation"),
                "SHOT_TYPE": shot.get("details", {}).get("shotType"),

                # Player info
                "SHOT_PLAYER_ID": shot.get("shotPlayer", {}).get("id"),
                "SHOT_PLAYER_DISPLAY_NAME": shot.get("shotPlayer", {}).get("displayName"),
                "SHOT_PLAYER_FULL_NAME": shot.get("shotPlayer", {}).get("fullname"),
                "SHOT_PLAYER_FIRST_NAME": shot.get("shotPlayer", {}).get("firstname"),
                "SHOT_PLAYER_LAST_NAME": shot.get("shotPlayer", {}).get("surname"),

                # Squad info
                "SHOT_SQUAD_ID": shot.get("shotPlayer", {}).get("squad", {}).get("id"),
                "SHOT_SQUAD_NAME": shot.get("shotPlayer", {}).get("squad", {}).get("name"),
                "SHOT_SQUAD_CODE": shot.get("shotPlayer", {}).get("squad", {}).get("code"),

                # Goal info (if applicable)
                "IS_GOAL": shot.get("goal") is not None,
                "GOAL_PLAYER_ID": shot.get("goal", {}).get("goalPlayer", {}).get("id") if shot.get("goal") else None,
                "GOAL_PLAYER_DISPLAY_NAME": shot.get("goal", {}).get("goalPlayer", {}).get("displayName") if shot.get("goal") else None,

                # Assist info
                "ASSIST_1_PLAYER_ID": shot.get("goal", {}).get("assistPlayers", [{}])[0].get("id") if shot.get("goal", {}) and shot.get("goal", {}).get("assistPlayers") else None,
                "ASSIST_1_PLAYER_DISPLAY_NAME": shot.get("goal", {}).get("assistPlayers", [{}])[0].get("displayName") if shot.get("goal", {}) and shot.get("goal", {}).get("assistPlayers") else None,
                "ASSIST_2_PLAYER_ID": shot.get("goal", {}).get("assistPlayers", [{}])[1].get("id") if shot.get("goal", {}) and len(shot.get("goal", {}).get("assistPlayers", [])) > 1 else None,
                "ASSIST_2_PLAYER_DISPLAY_NAME": shot.get("goal", {}).get("assistPlayers", [{}])[1].get("displayName") if shot.get("goal", {}) and len(shot.get("goal", {}).get("assistPlayers", [])) > 1 else None,
            }
            processed_shots.append(shot_data)

        return processed_shots

    async def collect_match_faceoffs(self, client: httpx.AsyncClient, match_id: str) -> List[Dict]:
        """Collect and process match faceoffs data"""
        data = await self.fetch_json(client, f"/matches/{match_id}/faceoffs")
        if not data or "faceOffs" not in data:
            return []

        processed_faceoffs = []
        for faceoff in data["faceOffs"]:
            faceoff_data = {
                "MATCH_ID": match_id,
                "MATCH_TRX_ID": faceoff.get("matchTrxId"),
                "PERIOD": faceoff.get("period"),
                "PERIOD_SECS": faceoff.get("periodSecs"),
                "PERIOD_DISPLAY": faceoff.get("periodDisplay"),
                "PERIOD_TIME": faceoff.get("periodTime"),
                "REMAINING_SECS": faceoff.get("remainingSecs"),
                "REMAINING_DISPLAY": faceoff.get("remainingDisplay"),
                "REMAINING_TIME": faceoff.get("remainingTime"),

                # Home player
                "HOME_PLAYER_ID": faceoff.get("homePlayer", {}).get("id"),
                "HOME_PLAYER_DISPLAY_NAME": faceoff.get("homePlayer", {}).get("displayName"),
                "HOME_PLAYER_FULL_NAME": faceoff.get("homePlayer", {}).get("fullname"),
                "HOME_SQUAD_ID": faceoff.get("homePlayer", {}).get("squad", {}).get("id"),
                "HOME_SQUAD_NAME": faceoff.get("homePlayer", {}).get("squad", {}).get("name"),
                "HOME_SQUAD_CODE": faceoff.get("homePlayer", {}).get("squad", {}).get("code"),

                # Away player
                "AWAY_PLAYER_ID": faceoff.get("awayPlayer", {}).get("id"),
                "AWAY_PLAYER_DISPLAY_NAME": faceoff.get("awayPlayer", {}).get("displayName"),
                "AWAY_PLAYER_FULL_NAME": faceoff.get("awayPlayer", {}).get("fullname"),
                "AWAY_SQUAD_ID": faceoff.get("awayPlayer", {}).get("squad", {}).get("id"),
                "AWAY_SQUAD_NAME": faceoff.get("awayPlayer", {}).get("squad", {}).get("name"),
                "AWAY_SQUAD_CODE": faceoff.get("awayPlayer", {}).get("squad", {}).get("code"),

                # Winner
                "WINNER_PLAYER_ID": faceoff.get("winnerPlayer", {}).get("id"),
                "WINNER_PLAYER_DISPLAY_NAME": faceoff.get("winnerPlayer", {}).get("displayName"),
                "WINNER_SQUAD_ID": faceoff.get("winnerPlayer", {}).get("squad", {}).get("id"),
                "WINNER_SQUAD_CODE": faceoff.get("winnerPlayer", {}).get("squad", {}).get("code"),
            }
            processed_faceoffs.append(faceoff_data)

        return processed_faceoffs

    async def collect_match_penalties(self, client: httpx.AsyncClient, match_id: str) -> List[Dict]:
        """Collect and process match penalties data"""
        data = await self.fetch_json(client, f"/matches/{match_id}/penalties")
        if not data or "penalties" not in data:
            return []

        processed_penalties = []
        for penalty in data["penalties"]:
            penalty_data = {
                "MATCH_ID": match_id,
                "MATCH_TRX_ID": penalty.get("matchTrxId"),
                "PERIOD": penalty.get("period"),
                "PERIOD_SECS": penalty.get("periodSecs"),
                "PERIOD_DISPLAY": penalty.get("periodDisplay"),
                "PERIOD_TIME": penalty.get("periodTime"),
                "REMAINING_SECS": penalty.get("remainingSecs"),
                "REMAINING_DISPLAY": penalty.get("remainingDisplay"),
                "REMAINING_TIME": penalty.get("remainingTime"),

                # Penalty details
                "PENALTY_TYPE": penalty.get("details", {}).get("penaltyType"),
                "PENALTY_DURATION": penalty.get("details", {}).get("penaltyDuration"),
                "PENALTY_SEVERITY": penalty.get("details", {}).get("penaltySeverity"),

                # Player info
                "PENALTY_PLAYER_ID": penalty.get("penaltyPlayer", {}).get("id"),
                "PENALTY_PLAYER_DISPLAY_NAME": penalty.get("penaltyPlayer", {}).get("displayName"),
                "PENALTY_PLAYER_FULL_NAME": penalty.get("penaltyPlayer", {}).get("fullname"),
                "PENALTY_SQUAD_ID": penalty.get("penaltyPlayer", {}).get("squad", {}).get("id"),
                "PENALTY_SQUAD_NAME": penalty.get("penaltyPlayer", {}).get("squad", {}).get("name"),
                "PENALTY_SQUAD_CODE": penalty.get("penaltyPlayer", {}).get("squad", {}).get("code"),
            }
            processed_penalties.append(penalty_data)

        return processed_penalties

    async def collect_match_player_statistics(self, client: httpx.AsyncClient, match_id: str) -> List[Dict]:
        """Collect and process match player statistics"""
        data = await self.fetch_json(client, f"/matches/{match_id}/statistics/players")
        if not data or "squads" not in data:
            return []

        processed_stats = []
        for squad in data["squads"]:
            squad_info = {
                "SQUAD_ID": squad.get("id"),
                "SQUAD_NAME": squad.get("name"),
                "SQUAD_CODE": squad.get("code"),
                "SQUAD_NICKNAME": squad.get("nickname")
            }

            for player in squad.get("players", []):
                player_data = {
                    "MATCH_ID": match_id,
                    "PLAYER_ID": player.get("id"),
                    "DISPLAY_NAME": player.get("displayName"),
                    "FIRST_NAME": player.get("firstName"),
                    "LAST_NAME": player.get("lastName"),
                    "JERSEY_NUMBER": player.get("jerseyNumber"),
                    **squad_info
                }

                # Add all statistics
                for stat in player.get("statistics", []):
                    stat_code = stat.get("code", "").upper()
                    player_data[stat_code] = stat.get("value", 0)
                    player_data[f"{stat_code}_DISPLAY"] = stat.get("valueDisplay", "")

                processed_stats.append(player_data)

        return processed_stats

    async def collect_match_team_statistics(self, client: httpx.AsyncClient, match_id: str) -> List[Dict]:
        """Collect and process match team statistics"""
        data = await self.fetch_json(client, f"/matches/{match_id}/statistics/squads")
        if not data or "squads" not in data:
            return []

        processed_stats = []
        for squad in data["squads"]:
            for stat in squad.get("statistics", []):
                stat_data = {
                    "MATCH_ID": match_id,
                    "SQUAD_ID": squad.get("id"),
                    "SQUAD_NAME": squad.get("name"),
                    "SQUAD_CODE": squad.get("code"),
                    "SQUAD_NICKNAME": squad.get("nickname"),
                    "STAT_ID": stat.get("id"),
                    "STAT_CODE": stat.get("code", "").upper(),
                    "STAT_NAME": stat.get("name"),
                    "STAT_NAME_PLURAL": stat.get("namePlural"),
                    "VALUE": stat.get("value", 0),
                    "VALUE_DISPLAY": stat.get("valueDisplay", "")
                }
                processed_stats.append(stat_data)

        return processed_stats

    async def collect_season_player_statistics(self, client: httpx.AsyncClient, season_id: str) -> List[Dict]:
        """Collect and process season player statistics"""
        data = await self.fetch_json(client, f"/leagues/{self.league_id}/levels/{self.level_id}/seasons/{season_id}/statistics/players")
        if not data or "squads" not in data:
            return []

        processed_stats = []
        for squad in data["squads"]:
            squad_info = {
                "SQUAD_ID": squad.get("id"),
                "SQUAD_NAME": squad.get("name"),
                "SQUAD_CODE": squad.get("code"),
                "SQUAD_NICKNAME": squad.get("nickname")
            }

            for player in squad.get("players", []):
                player_data = {
                    "SEASON_ID": season_id,
                    "PLAYER_ID": player.get("id"),
                    "DISPLAY_NAME": player.get("displayName"),
                    "FIRST_NAME": player.get("firstName"),
                    "LAST_NAME": player.get("lastName"),
                    "JERSEY_NUMBER": player.get("jerseyNumber"),
                    **squad_info
                }

                # Add all statistics
                for stat in player.get("statistics", []):
                    stat_code = stat.get("code", "").upper()
                    player_data[stat_code] = stat.get("value", 0)
                    player_data[f"{stat_code}_DISPLAY"] = stat.get("valueDisplay", "")

                processed_stats.append(player_data)

        return processed_stats

    async def collect_season_team_statistics(self, client: httpx.AsyncClient, season_id: str) -> List[Dict]:
        """Collect and process season team statistics"""
        data = await self.fetch_json(client, f"/leagues/{self.league_id}/levels/{self.level_id}/seasons/{season_id}/statistics/squads")
        if not data or "squads" not in data:
            return []

        processed_stats = []
        for squad in data["squads"]:
            for stat in squad.get("statistics", []):
                stat_data = {
                    "SEASON_ID": season_id,
                    "SQUAD_ID": squad.get("id"),
                    "SQUAD_NAME": squad.get("name"),
                    "SQUAD_CODE": squad.get("code"),
                    "SQUAD_NICKNAME": squad.get("nickname"),
                    "STAT_ID": stat.get("id"),
                    "STAT_CODE": stat.get("code", "").upper(),
                    "STAT_NAME": stat.get("name"),
                    "STAT_NAME_PLURAL": stat.get("namePlural"),
                    "VALUE": stat.get("value", 0),
                    "VALUE_DISPLAY": stat.get("valueDisplay", "")
                }
                processed_stats.append(stat_data)

        return processed_stats

# Test functions for debugging
async def test_match_data(match_id: str = "786698274"):
    """Test match data collection with populated match"""
    print(f"🧪 Testing match data collection for match {match_id}")

    collector = NLLDataCollector()

    async with httpx.AsyncClient(timeout=30) as client:
        # Initialize discovery
        await collector.initialize_discovery(client)

        # Test each match endpoint
        shots = await collector.collect_match_shots(client, match_id)
        print(f"📊 Shots: {len(shots)} records")

        faceoffs = await collector.collect_match_faceoffs(client, match_id)
        print(f"📊 Faceoffs: {len(faceoffs)} records")

        penalties = await collector.collect_match_penalties(client, match_id)
        print(f"📊 Penalties: {len(penalties)} records")

        player_stats = await collector.collect_match_player_statistics(client, match_id)
        print(f"📊 Player stats: {len(player_stats)} records")

        team_stats = await collector.collect_match_team_statistics(client, match_id)
        print(f"📊 Team stats: {len(team_stats)} records")

        # Save samples for inspection
        if shots:
            pd.DataFrame(shots[:5]).to_csv("sample_shots.csv", index=False)
            print("💾 Sample shots saved to sample_shots.csv")

        if faceoffs:
            pd.DataFrame(faceoffs[:5]).to_csv("sample_faceoffs.csv", index=False)
            print("💾 Sample faceoffs saved to sample_faceoffs.csv")

if __name__ == "__main__":
    asyncio.run(test_match_data())
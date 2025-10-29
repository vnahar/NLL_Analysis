#!/usr/bin/env python3
"""
NLL Data Pipeline V2 - Clean rebuild using modular data collection functions

This version uses the tested modular functions from nll_data_collectors.py
to ensure proper data flattening and no empty columns.

Usage:
  python nll_pipeline_v2.py

Output:
  - CSV files in out_csv/
  - NLL_Data_2020_2024.xlsx (final Excel workbook)
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import httpx
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# Import our modular data collection functions
from nll_data_collectors import NLLDataCollector

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
out_dir = Path("out_csv")
out_dir.mkdir(exist_ok=True)
limiter = AsyncLimiter(15, 1)  # 15 requests per second

class NLLPipelineV2:
    """Clean NLL data pipeline using modular functions"""

    def __init__(self):
        self.collector = NLLDataCollector()
        self.all_match_ids = []

    async def collect_all_data(self):
        """Main data collection orchestrator"""
        logger.info("🚀 Starting NLL data collection with modular functions...")

        async with httpx.AsyncClient(timeout=60) as client:
            # Initialize discovery data
            if not await self.collector.initialize_discovery(client):
                logger.error("Failed to initialize discovery data")
                return

            # Step 1: Collect career statistics (these work well)
            await self.collect_career_data(client)

            # Step 2: Get all match IDs from schedules
            await self.collect_match_ids(client)

            # Step 3: Collect match-level data (shots, faceoffs, penalties, stats)
            await self.collect_match_data(client)

            # Step 4: Collect season-level data
            await self.collect_season_data(client)

    async def collect_career_data(self, client: httpx.AsyncClient):
        """Collect career statistics data"""
        logger.info("📊 Collecting career statistics...")

        # Regular season career stats
        career_reg = await self.collector.collect_career_statistics(client, "REG")
        if career_reg:
            df = pd.DataFrame(career_reg)
            df.to_csv(out_dir / "career_players_reg.csv", index=False)
            logger.info(f"✅ Saved {len(career_reg)} regular season career records")

        # Playoff career stats
        career_post = await self.collector.collect_career_statistics(client, "POST")
        if career_post:
            df = pd.DataFrame(career_post)
            df.to_csv(out_dir / "career_players_post.csv", index=False)
            logger.info(f"✅ Saved {len(career_post)} playoff career records")

    async def collect_match_ids(self, client: httpx.AsyncClient):
        """Collect all match IDs from schedules"""
        logger.info("🗓️ Collecting match IDs from schedules...")

        for season in self.collector.seasons:
            season_id = str(season["id"])
            season_name = season.get("competitionName", f"Season {season_id}")
            logger.info(f"  📅 Processing {season_name}")

            # Get schedule
            schedule = await self.collector.fetch_json(
                client,
                f"/leagues/{self.collector.league_id}/levels/{self.collector.level_id}/seasons/{season_id}/schedule"
            )

            if isinstance(schedule, dict) and "phases" in schedule:
                for phase in schedule["phases"]:
                    for week in phase.get("weeks", []):
                        for match in week.get("matches", []):
                            match_id = str(match.get("matchId") or match.get("id"))
                            if match_id and match_id not in self.all_match_ids:
                                self.all_match_ids.append(match_id)

        logger.info(f"✅ Found {len(self.all_match_ids)} total matches")

    async def collect_match_data(self, client: httpx.AsyncClient):
        """Collect all match-level data using populated matches"""
        logger.info("🏒 Collecting match-level data...")

        # Load populated matches to prioritize matches with data
        populated_matches_file = Path("populated_matches.json")
        populated_match_ids = []
        if populated_matches_file.exists():
            import json
            with open(populated_matches_file) as f:
                populated_data = json.load(f)
                populated_match_ids = [m["match_id"] for m in populated_data]

        # Prioritize populated matches, then process remaining
        prioritized_matches = populated_match_ids + [m for m in self.all_match_ids if m not in populated_match_ids]

        # Collect data from matches (limit to prevent overwhelming)
        all_shots = []
        all_faceoffs = []
        all_penalties = []
        all_player_stats = []
        all_team_stats = []

        matches_processed = 0
        matches_with_data = 0

        for match_id in prioritized_matches[:200]:  # Process first 200 matches
            matches_processed += 1

            if matches_processed % 20 == 0:
                logger.info(f"  Processed {matches_processed} matches, {matches_with_data} with data")

            # Collect match data
            shots = await self.collector.collect_match_shots(client, match_id)
            faceoffs = await self.collector.collect_match_faceoffs(client, match_id)
            penalties = await self.collector.collect_match_penalties(client, match_id)
            player_stats = await self.collector.collect_match_player_statistics(client, match_id)
            team_stats = await self.collector.collect_match_team_statistics(client, match_id)

            data_count = len(shots) + len(faceoffs) + len(penalties) + len(player_stats) + len(team_stats)

            if data_count > 0:
                matches_with_data += 1
                all_shots.extend(shots)
                all_faceoffs.extend(faceoffs)
                all_penalties.extend(penalties)
                all_player_stats.extend(player_stats)
                all_team_stats.extend(team_stats)

            # Stop early if we have enough data
            if len(all_player_stats) > 10000 and len(all_shots) > 5000:
                logger.info(f"  ✅ Collected sufficient data, stopping early")
                break

        # Save match data
        if all_shots:
            pd.DataFrame(all_shots).to_csv(out_dir / "shots.csv", index=False)
            logger.info(f"✅ Saved {len(all_shots)} shot records")

        if all_faceoffs:
            pd.DataFrame(all_faceoffs).to_csv(out_dir / "faceoffs.csv", index=False)
            logger.info(f"✅ Saved {len(all_faceoffs)} faceoff records")

        if all_penalties:
            pd.DataFrame(all_penalties).to_csv(out_dir / "penalties.csv", index=False)
            logger.info(f"✅ Saved {len(all_penalties)} penalty records")

        if all_player_stats:
            pd.DataFrame(all_player_stats).to_csv(out_dir / "player_stats_match.csv", index=False)
            logger.info(f"✅ Saved {len(all_player_stats)} player match stat records")

        if all_team_stats:
            pd.DataFrame(all_team_stats).to_csv(out_dir / "team_stats_match.csv", index=False)
            logger.info(f"✅ Saved {len(all_team_stats)} team match stat records")

        logger.info(f"🏒 Match data collection complete: {matches_with_data}/{matches_processed} matches had data")

    async def collect_season_data(self, client: httpx.AsyncClient):
        """Collect season-level statistics"""
        logger.info("📈 Collecting season-level data...")

        all_player_season_stats = []
        all_team_season_stats = []

        for season in self.collector.seasons:
            season_id = str(season["id"])
            season_name = season.get("competitionName", f"Season {season_id}")
            logger.info(f"  📅 Processing {season_name}")

            # Collect season player stats
            player_stats = await self.collector.collect_season_player_statistics(client, season_id)
            all_player_season_stats.extend(player_stats)

            # Collect season team stats
            team_stats = await self.collector.collect_season_team_statistics(client, season_id)
            all_team_season_stats.extend(team_stats)

        # Save season data
        if all_player_season_stats:
            pd.DataFrame(all_player_season_stats).to_csv(out_dir / "player_stats_season.csv", index=False)
            logger.info(f"✅ Saved {len(all_player_season_stats)} player season stat records")

        if all_team_season_stats:
            pd.DataFrame(all_team_season_stats).to_csv(out_dir / "team_stats_season.csv", index=False)
            logger.info(f"✅ Saved {len(all_team_season_stats)} team season stat records")

    def create_excel_workbook(self):
        """Create the final Excel workbook with all data"""
        logger.info("📊 Creating Excel workbook...")

        # Define the expected files and their sheet names
        files_to_sheets = {
            "career_players_reg.csv": "Career Players Reg",
            "career_players_post.csv": "Career Players Post",
            "player_stats_match.csv": "Player Stats Match",
            "player_stats_season.csv": "Player Stats Season",
            "team_stats_match.csv": "Team Stats Match",
            "team_stats_season.csv": "Team Stats Season",
            "shots.csv": "Shots",
            "faceoffs.csv": "Faceoffs",
            "penalties.csv": "Penalties"
        }

        excel_file = Path("NLL_Data_2020_2024_V2.xlsx")

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for filename, sheet_name in files_to_sheets.items():
                file_path = out_dir / filename

                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)

                        # Convert column names to uppercase for consistency
                        df.columns = [col.upper() for col in df.columns]

                        # Write to Excel
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        logger.info(f"  ✅ Added {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        logger.error(f"  ❌ Failed to process {filename}: {e}")
                else:
                    logger.warning(f"  ⚠️ File not found: {filename}")

        logger.info(f"✅ Excel workbook created: {excel_file}")

    async def run_pipeline(self):
        """Run the complete pipeline"""
        try:
            await self.collect_all_data()
            self.create_excel_workbook()
            logger.info("🎉 Pipeline completed successfully!")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

async def main():
    """Main entry point"""
    pipeline = NLLPipelineV2()
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
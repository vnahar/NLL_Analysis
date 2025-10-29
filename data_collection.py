"""
ChampionData → CSV (2020–2024), one CSV per logical table.

Install:
  python -m pip install "httpx[http2]" aiolimiter tenacity pandas pyarrow

Run:
  export NLL_API_TOKEN="YOUR_BEARER_TOKEN"
  python data_collection.py
"""

import os
import asyncio
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, AsyncIterator, Tuple

import httpx
import pandas as pd
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# CONFIG — set these first
# =========================
BASE_URL = "https://api.nll.championdata.io/v1"
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjRYa0U1VDZRZkFpMFFaZnRRaFQweiJ9.eyJodHRwczovL3NjaGVtYS5jaGFtcGlvbmRhdGEuY29tLmF1L25hbWUiOiJKdWRlIEZlcm5hbmRlcyIsImh0dHBzOi8vc2NoZW1hLmNoYW1waW9uZGF0YS5jb20uYXUvZW1haWwiOiJqZmVybmFuZGVzQGFsdHNwb3J0c2RhdGEuY29tIiwiaXNzIjoiaHR0cHM6Ly9jaGFtcGlvbmRhdGEuYXUuYXV0aDAuY29tLyIsInN1YiI6ImF1dGgwfDY4MWE2OWMyOWY2MjU4NjkzMDBiZTM4OSIsImF1ZCI6Imh0dHBzOi8vYXBpLm5sbC5jaGFtcGlvbmRhdGEuaW8vIiwiaWF0IjoxNzYxMjY5NTU5LCJleHAiOjE3NjEyNzY3NTksInNjb3BlIjoiIiwiYXpwIjoibjhyNlJmbmVxbFNLTXdWRUh0bU56WE5YWGlsUVE4MGYiLCJwZXJtaXNzaW9ucyI6WyJhdXRoOmJhc2ljIiwiZW52OnByb2QiLCJlbnY6cHJvZC1zYW5kYm94IiwicmVhZDpkZWZhdWx0IiwidXNhZ2U6bG93Il19.gbEq0zg6r5VMk0avKo6MlSEvz9AwViCyOUceRVTzvEdOZzImCGyMHTW603aeoeYSLnDDVzwf0vLqV6qPPJLaZpp007RqgMPLX9SQJOOnqGzDoafgH7rjvNfNqGDQvMBIhY11a4oJ_D2ZNW9joF5V2CXKiRiljGcQBOafdfNPOZ_jMLb1yYDls7EFjApn2lQ6Byb9zoJf8VIpIlzoVvANwLiMfi7f5q_KzK6soQqVfF44KS4VodBlop4hwaporsE-wWk5fM60smSrgDd5PCEwZ-9qKMDJYrWhcEz1ldo2Df4lu8fpxll3brejzZaSUwJmgUVw4CNcml7m23L-GIplEA"
if not TOKEN:
    raise RuntimeError("Set NLL_API_TOKEN environment variable with your Bearer token.")

AUTH_HEADER = {"Authorization": f"Bearer {TOKEN}", "accept": "application/json"}

# API IDs - will be discovered dynamically
LEAGUE_ID = None  # Will be set via discovery
LEVEL_ID = None   # Will be set via discovery

# Time window (inclusive)
SINCE = dt.datetime(2020, 1, 1)
UNTIL = dt.datetime(2024, 12, 31, 23, 59, 59)

# Concurrency / rate-limiting (conservative settings)
CONCURRENCY = 75      # Reduced for stability
RPS_LIMIT   = 15      # Conservative rate limit
TIMEOUT_S   = 30.0    # Allow more time for responses
MAX_PAGES   = 50      # Safety limit for pagination

# Output directory
OUT = Path("out_csv")
OUT.mkdir(parents=True, exist_ok=True)

# Logical tables → file paths (canonical schema)
TABLES = {
    "seasons": OUT / "seasons.csv",
    "teams": OUT / "teams.csv",
    "players": OUT / "players.csv",
    "player_stints": OUT / "player_stints.csv",
    "schedule": OUT / "schedule.csv",
    "scores_match": OUT / "scores_match.csv",
    "scores_period": OUT / "scores_period.csv",
    "team_stats_match": OUT / "team_stats_match.csv",
    "team_stats_match_flat": OUT / "team_stats_match_flat.csv",
    "player_stats_match": OUT / "player_stats_match.csv",
    "player_stats_match_flat": OUT / "player_stats_match_flat.csv",
    "team_stats_season": OUT / "team_stats_season.csv",
    "player_stats_season": OUT / "player_stats_season.csv",
    "shots": OUT / "shots.csv",
    "penalties": OUT / "penalties.csv",
    "faceoffs": OUT / "faceoffs.csv",
    "standings": OUT / "standings.csv",
    "career_players_REG": OUT / "career_players_REG.csv",
    "career_players_POST": OUT / "career_players_POST.csv",
}

# =========================
# Infra: client, limiter, csv writer
# =========================
sema = asyncio.Semaphore(CONCURRENCY)
limiter = AsyncLimiter(max_rate=RPS_LIMIT, time_period=1)

def _append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append df to CSV, writing header only if file doesn't exist."""
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    # Reorder columns for consistency (optional: sort for stable schema)
    cols = sorted(df.columns)
    df[cols].to_csv(path, mode="a", header=write_header, index=False)

@retry(wait=wait_exponential_jitter(initial=0.5, max=8), stop=stop_after_attempt(3))
async def _get_raw(client: httpx.AsyncClient, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET JSON from absolute URL, with rate limiting & retries."""
    start_time = asyncio.get_event_loop().time()
    async with sema:
        async with limiter:
            logger.debug(f"Requesting: {url}")
            r = await client.get(url, headers=AUTH_HEADER, params=params, timeout=TIMEOUT_S)
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.debug(f"Request completed in {elapsed:.2f}s: {url} -> {r.status_code}")
    r.raise_for_status()
    return r.json()

async def _get_json(client: httpx.AsyncClient, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET JSON from path (joined to BASE_URL)."""
    url = f"{BASE_URL}{path}"
    return await _get_raw(client, url, params)

def _norm(obj: Union[List, Dict]) -> pd.DataFrame:
    """Normalize JSON into a flat DataFrame (best-effort)."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame()
        return pd.json_normalize(obj)
    return pd.json_normalize([obj])

def _within_window(when: Optional[str]) -> bool:
    if not when:
        return False
    ts = pd.to_datetime(when, utc=True, errors="coerce")
    if pd.isna(ts):
        return False
    ts = ts.tz_convert(None) if ts.tzinfo else ts
    return SINCE <= ts <= UNTIL

async def _paged_items(
    client: httpx.AsyncClient,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    container_hint: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Iterate all items across pages.
    - Supports dict payloads with `links.next`
    - Detects list container via `container_hint` or common keys.
    """
    url = f"{BASE_URL}{path}"
    q = dict(params or {})
    page_count = 0
    total_items = 0

    logger.info(f"Starting pagination for: {path}")

    while page_count < MAX_PAGES:
        page_count += 1
        logger.debug(f"Fetching page {page_count} from: {url}")

        try:
            payload = await _get_raw(client, url, q if url.startswith(BASE_URL) else None)
        except Exception as e:
            logger.warning(f"Failed to fetch page {page_count} for {path}: {e}")
            break

        # If top-level is a list — easy
        if isinstance(payload, list):
            items = payload
            next_url = None
        else:
            # Dict — find the array container
            keys = (container_hint,) if container_hint else (
                "players", "matches", "squads", "items", "data", "rows"
            )
            items = None
            for k in keys:
                if isinstance(payload, dict) and isinstance(payload.get(k), list):
                    items = payload[k]
                    break
            if items is None:
                items = []
            # Next link (absolute)
            links = payload.get("links") if isinstance(payload, dict) else None
            next_url = links.get("next") if isinstance(links, dict) else None

        logger.debug(f"Page {page_count}: {len(items)} items found")
        total_items += len(items)

        for it in items:
            yield it

        # Follow absolute next URL if present
        if next_url:
            logger.debug(f"Following next URL: {next_url}")
            url = next_url
            q = {}
            continue

        # Fallback manual pagination: stop if we got fewer than limit
        limit = (params or {}).get("limit", 100)
        if isinstance(items, list) and len(items) >= int(limit):
            q = dict(q or params or {})
            page = int(q.get("page", 1)) + 1
            q["page"] = page
            url = f"{BASE_URL}{path}"
            logger.debug(f"Manual pagination: next page {page}")
            continue
        break

    logger.info(f"Pagination complete for {path}: {total_items} total items across {page_count} pages")

# =========================
# Discovery (leagues, levels, seasons, teams, players)
# =========================

async def discover_league_level(client: httpx.AsyncClient) -> Tuple[str, str]:
    """Discover NLL league and level IDs dynamically."""
    global LEAGUE_ID, LEVEL_ID

    if LEAGUE_ID and LEVEL_ID:
        return LEAGUE_ID, LEVEL_ID

    logger.info("Discovering NLL league and level IDs...")

    # Get leagues
    leagues = await _get_json(client, "/leagues")
    if isinstance(leagues, list):
        nll_league = next((lg for lg in leagues if lg.get("code") == "NLL"), leagues[0])
    else:
        nll_league = leagues

    LEAGUE_ID = str(nll_league["id"])
    logger.info(f"Found NLL league ID: {LEAGUE_ID}")

    # Get levels for this league
    levels = await _get_json(client, f"/leagues/{LEAGUE_ID}/levels")
    if isinstance(levels, dict) and "levels" in levels:
        levels = levels["levels"]

    LEVEL_ID = str(levels[0]["id"]) if levels else "1"
    logger.info(f"Found level ID: {LEVEL_ID}")

    return LEAGUE_ID, LEVEL_ID
async def fetch_seasons(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    logger.info("Fetching seasons data...")
    seasons_payload = await _get_json(client, f"/leagues/{LEAGUE_ID}/levels/{LEVEL_ID}/seasons")

    # Extract seasons array from wrapper object
    if isinstance(seasons_payload, dict) and "seasons" in seasons_payload:
        seasons = seasons_payload["seasons"]
    else:
        seasons = seasons_payload if isinstance(seasons_payload, list) else []

    logger.info(f"Found {len(seasons)} total seasons")

    keep = []
    for s in seasons:
        # Use startYear field from API schema
        start_year = s.get("startYear")
        end_year = s.get("endYear")

        # Filter to 2020-2025 range based on start year (include current season)
        if start_year is not None and 2020 <= int(start_year) <= 2025:
            keep.append(s)
        elif end_year is not None and 2020 <= int(end_year) <= 2025:
            keep.append(s)

    logger.info(f"Filtered to {len(keep)} seasons in 2020-2025 range")

    df = _norm(keep)
    if not df.empty:
        df["leagueId"] = LEAGUE_ID
        df["levelId"]  = LEVEL_ID
        _append_csv(df, TABLES["seasons"])
        logger.info(f"Saved {len(df)} seasons to CSV")
    return keep

async def fetch_teams_players_rosters(client: httpx.AsyncClient, season_id: str) -> None:
    # Teams (squads) - handle potential 404s from probe results
    try:
        squads_payload = await _get_json(client, f"/leagues/{LEAGUE_ID}/levels/{LEVEL_ID}/seasons/{season_id}/squads")
        # Extract squads from response structure
        if isinstance(squads_payload, dict) and "squads" in squads_payload:
            squads = squads_payload["squads"]
        else:
            squads = squads_payload if isinstance(squads_payload, list) else []

        df_teams = _norm(squads)
        if not df_teams.empty:
            df_teams["seasonId"] = season_id
            _append_csv(df_teams, TABLES["teams"])
    except Exception as e:
        print(f"Warning: Could not fetch squads for season {season_id}: {e}")
        squads = []

    # Players
    try:
        players_rows = [p async for p in _paged_items(
            client,
            f"/leagues/{LEAGUE_ID}/levels/{LEVEL_ID}/seasons/{season_id}/players",
            params={"page": 1, "limit": 100},
            container_hint="players",
        )]
        df_players = _norm(players_rows)
        if not df_players.empty:
            df_players["seasonId"] = season_id
            _append_csv(df_players, TABLES["players"])
    except Exception as e:
        print(f"Warning: Could not fetch players for season {season_id}: {e}")

    # Rosters (link of persons/players to squads) - skip if no squads
    if not squads:
        print(f"Skipping roster fetch for season {season_id} - no squads available")
        return

    tasks = []
    for sq in squads:
        squad_id = sq.get("squadId") or sq.get("id")
        if not squad_id:
            continue
        tasks.append(_get_json(
            client,
            f"/leagues/{LEAGUE_ID}/levels/{LEVEL_ID}/seasons/{season_id}/squads/{squad_id}/persons"
        ))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        rows = []
        for res in results:
            if isinstance(res, Exception):
                print(f"Warning: Roster fetch failed: {res}")
                continue
            if not res:
                continue
            # Handle different response structures
            persons = res if isinstance(res, list) else res.get("persons", [])
            for row in persons:
                if isinstance(row, dict):
                    row["seasonId"] = season_id
                    rows.append(row)
        if rows:
            _append_csv(_norm(rows), TABLES["rosters"])

# =========================
# Schedule (filter to 2020–2024)
# =========================
async def fetch_schedule(client: httpx.AsyncClient, season_id: str) -> List[Dict[str, Any]]:
    """Fetch schedule with proper parsing of nested phases/weeks/matches structure."""
    try:
        # Get the full schedule response (not paginated - single endpoint)
        schedule_data = await _get_json(
            client,
            f"/leagues/{LEAGUE_ID}/levels/{LEVEL_ID}/seasons/{season_id}/schedule"
        )

        if not schedule_data:
            logger.warning(f"❌ No schedule data returned for season {season_id}")
            return []

        # Extract matches from nested structure: phases[].weeks[].matches[]
        all_matches = []
        phases = schedule_data.get("phases", [])

        for phase in phases:
            weeks = phase.get("weeks", [])
            for week in weeks:
                matches = week.get("matches", [])
                for match in matches:
                    # Add week info to match
                    match["weekNumber"] = week.get("number")
                    match["weekName"] = week.get("name")
                    match["phaseNumber"] = phase.get("number")
                    all_matches.append(match)

        logger.info(f"Found {len(all_matches)} total matches in schedule for season {season_id}")

        keep = []
        for m in all_matches:
            # Extract date from nested structure
            date_obj = m.get("date", {})
            when = (date_obj.get("utcMatchStart") or
                   date_obj.get("startDate") or
                   m.get("start") or m.get("startTime"))

            # Normalize match data with canonical fields
            squads = m.get("squads", {})
            home_squad = squads.get("home", {})
            away_squad = squads.get("away", {})
            status = m.get("status", {})
            venue = m.get("venue", {})

            normalized_match = {
                'matchId': str(m.get('id')),
                'seasonId': season_id,
                'weekNumber': m.get('weekNumber'),
                'weekName': m.get('weekName'),
                'phaseNumber': m.get('phaseNumber'),
                'startTimeUTC': when,
                'statusId': status.get('id'),
                'statusName': status.get('name'),
                'statusCode': status.get('code'),
                'homeSquadId': str(home_squad.get('id', '')),
                'homeSquadCode': home_squad.get('code'),
                'homeSquadName': home_squad.get('displayName'),
                'homeScore': home_squad.get('score', {}).get('goals'),
                'awaySquadId': str(away_squad.get('id', '')),
                'awaySquadCode': away_squad.get('code'),
                'awaySquadName': away_squad.get('displayName'),
                'awayScore': away_squad.get('score', {}).get('goals'),
                'venueId': venue.get('id'),
                'venueName': venue.get('name'),
                'venueCode': venue.get('code'),
                'winningSquadId': str(m.get('winningSquadId', ''))
            }
            keep.append(normalized_match)

        if keep:
            df = pd.DataFrame(keep)
            _append_csv(df, TABLES["schedule"])
            logger.info(f"✅ Saved {len(keep)} matches to schedule for season {season_id}")

        return keep
    except Exception as e:
        logger.warning(f"Could not fetch schedule for season {season_id}: {e}")
        return []

# =========================
# Per-match fetch bundle
# =========================
async def fetch_match_bundle(client: httpx.AsyncClient, match: Dict[str, Any]) -> None:
    match_id = match.get("matchId") or match.get("id")
    if not match_id:
        return

    async def safe(name: str, coro):
        try:
            data = await coro
            return (name, data)
        except Exception:
            return (name, None)

    calls = [
        safe("score", _get_json(client, f"/matches/{match_id}/score")),
        safe("team_stats", _get_json(client, f"/matches/{match_id}/statistics/squads")),
        safe("player_stats", _get_json(client, f"/matches/{match_id}/statistics/players")),
        safe("shots", _get_json(client, f"/matches/{match_id}/shots")),
        safe("penalties", _get_json(client, f"/matches/{match_id}/penalties")),
        safe("faceoffs", _get_json(client, f"/matches/{match_id}/faceoffs")),
    ]
    results = await asyncio.gather(*calls)

    name_to_data = {k: v for k, v in results}

    # Score → split into match-level row and per-period rows if present
    score = name_to_data.get("score")
    if score:
        if isinstance(score, dict):
            # Match-level (top fields)
            df_score_match = _norm({k: v for k, v in score.items() if k != "periodScores"})
            df_score_match["matchId"] = match_id
            _append_csv(df_score_match, TABLES["scores_match"])

            # Period-level breakdown
            periods = score.get("periodScores") or []
            for p in periods:
                p["matchId"] = match_id
            _append_csv(_norm(periods), TABLES["scores_period"])
        elif isinstance(score, list):
            # Some APIs return a list; write as match-level rows
            df_score_match = _norm(score)
            if not df_score_match.empty:
                df_score_match["matchId"] = match_id
                _append_csv(df_score_match, TABLES["scores_match"])

    # Team stats (per match)
    team_stats = name_to_data.get("team_stats") or []
    if isinstance(team_stats, dict) and "squads" in team_stats:
        team_stats = team_stats["squads"]
    for r in team_stats if isinstance(team_stats, list) else []:
        r["matchId"] = match_id
    _append_csv(_norm(team_stats if isinstance(team_stats, list) else []), TABLES["team_stats_match"])

    # Player stats (per match)
    player_stats = name_to_data.get("player_stats") or []
    if isinstance(player_stats, dict) and "players" in player_stats:
        player_stats = player_stats["players"]
    for r in player_stats if isinstance(player_stats, list) else []:
        r["matchId"] = match_id
    _append_csv(_norm(player_stats if isinstance(player_stats, list) else []), TABLES["player_stats_match"])

    # Events (shots, penalties, faceoffs)
    for key, table in [("shots", "shots"), ("penalties", "penalties"), ("faceoffs", "faceoffs")]:
        items = name_to_data.get(key) or []
        # If wrapped (e.g., {"shots":[...]})
        if isinstance(items, dict):
            # pick the first list value
            for v in items.values():
                if isinstance(v, list):
                    items = v
                    break
        if isinstance(items, list):
            for r in items:
                r["matchId"] = match_id
            _append_csv(_norm(items), TABLES[table])

# =========================
# Season aggregates & standings
# =========================
async def fetch_season_aggregates(client: httpx.AsyncClient, season_id: str) -> None:
    # Team season stats (container often 'squads')
    try:
        squads_iter = _paged_items(
            client,
            f"/leagues/{LEAGUE_ID}/levels/{LEVEL_ID}/seasons/{season_id}/statistics/squads",
            params={"page": 1, "limit": 200},
            container_hint="squads",
        )
        squads = [s async for s in squads_iter]
        df = _norm(squads)
        if not df.empty:
            df["seasonId"] = season_id
            _append_csv(df, TABLES["team_stats_season"])
    except Exception as e:
        print(f"Warning: Could not fetch team stats for season {season_id}: {e}")

    # Player season stats (container 'players')
    try:
        players_iter = _paged_items(
            client,
            f"/leagues/{LEAGUE_ID}/levels/{LEVEL_ID}/seasons/{season_id}/statistics/players",
            params={"page": 1, "limit": 200},
            container_hint="players",
        )
        players = [p async for p in players_iter]
        df = _norm(players)
        if not df.empty:
            df["seasonId"] = season_id
            _append_csv(df, TABLES["player_stats_season"])
    except Exception as e:
        print(f"Warning: Could not fetch player stats for season {season_id}: {e}")

    # Standings (usually not paginated, but handle both)
    try:
        standings_payload = await _get_json(client, f"/leagues/{LEAGUE_ID}/levels/{LEVEL_ID}/seasons/{season_id}/standings")
        if isinstance(standings_payload, dict):
            # container guess
            for key in ("standings", "rows", "items", "data"):
                if isinstance(standings_payload.get(key), list):
                    df = _norm(standings_payload[key])
                    break
            else:
                df = _norm(standings_payload)
        else:
            df = _norm(standings_payload)
        if not df.empty:
            df["seasonId"] = season_id
            _append_csv(df, TABLES["standings"])
    except Exception as e:
        print(f"Warning: Could not fetch standings for season {season_id}: {e}")

async def fetch_career_statistics(client: httpx.AsyncClient) -> None:
    """
    Fetch career statistics for players split by regular season (REG) and postseason (POST).
    These provide career-long aggregates useful for priors or award narratives.
    """
    phases = ["REG", "POST"]

    for phase in phases:
        logger.info(f"📈 Collecting career statistics for {phase} phase...")
        try:
            players_iter = _paged_items(
                client,
                "/career/statistics/players",
                params={"phase": phase, "page": 1, "limit": 100},
                container_hint="players",
            )
            players = [p async for p in players_iter]

            if players:
                df = _norm(players)
                df["phase"] = phase  # Add phase identifier
                _append_csv(df, TABLES[f"career_players_{phase}"])
                logger.info(f"✅ Collected {len(df)} career player records for {phase} phase")
            else:
                logger.warning(f"⚠️  No career players found for {phase} phase")

        except Exception as e:
            logger.error(f"❌ Failed to fetch career statistics for {phase} phase: {e}")

# =========================
# Orchestration
# =========================
async def main():
    async with httpx.AsyncClient(http2=True) as client:
        # 0) Discover league and level IDs
        await discover_league_level(client)

        # 1) Discover seasons in 2020–2024 and write to seasons.csv
        seasons = await fetch_seasons(client)

        # 2) For each season: reference data + schedule
        for s in seasons:
            # Use correct field name from API schema (id, not seasonId)
            season_id = str(s.get("id"))
            if not season_id or season_id == "None":
                continue

            logger.info(f"🏒 Processing season {season_id} ({s.get('competitionName', 'Unknown')})")

            season_start = asyncio.get_event_loop().time()
            await fetch_teams_players_rosters(client, season_id)
            sched = await fetch_schedule(client, season_id)

            # 3) Fan out per-match bundles in polite batches (only if we have matches)
            if sched:
                logger.info(f"⚡ Processing {len(sched)} matches for season {season_id}")
                BATCH = 100  # Reduced batch size for better progress tracking
                for i in range(0, len(sched), BATCH):
                    chunk = sched[i:i+BATCH]
                    batch_start = asyncio.get_event_loop().time()
                    await asyncio.gather(*(fetch_match_bundle(client, m) for m in chunk))
                    batch_elapsed = asyncio.get_event_loop().time() - batch_start
                    logger.info(f"📊 Completed batch {i//BATCH + 1}/{(len(sched)-1)//BATCH + 1} ({len(chunk)} matches) in {batch_elapsed:.1f}s")
            else:
                logger.warning(f"❌ No matches found for season {season_id}")

            # 4) Season aggregates & standings
            await fetch_season_aggregates(client, season_id)

            season_elapsed = asyncio.get_event_loop().time() - season_start
            logger.info(f"✅ Season {season_id} completed in {season_elapsed:.1f}s")

        # 5) Career statistics (REG/POST splits) - collected once after all seasons
        await fetch_career_statistics(client)

def flatten_stats_to_columns(input_file: Path, output_file: Path, id_cols: List[str], stats_col: str = "statistics") -> None:
    """
    Post-process statistics CSV files to pivot statistics array into individual columns.

    Args:
        input_file: Path to input CSV with nested statistics
        output_file: Path to output flattened CSV
        id_cols: List of ID columns to preserve (e.g., ['matchId', 'squadId'])
        stats_col: Name of the statistics column (default: 'statistics')
    """
    if not input_file.exists():
        logger.warning(f"Input file {input_file} not found for flattening")
        return

    try:
        df = pd.read_csv(input_file)
        if df.empty:
            logger.warning(f"Input file {input_file} is empty")
            return

        logger.info(f"Flattening {len(df)} rows from {input_file.name}")

        flattened_rows = []
        for _, row in df.iterrows():
            # Start with ID columns
            flat_row = {col: row[col] for col in id_cols if col in row}

            # Parse and flatten statistics
            try:
                stats = row.get(stats_col, [])
                if isinstance(stats, str):
                    import ast
                    stats = ast.literal_eval(stats)

                # Convert statistics array to dictionary
                if isinstance(stats, list):
                    for stat in stats:
                        if isinstance(stat, dict) and 'code' in stat and 'value' in stat:
                            flat_row[stat['code']] = stat['value']

            except Exception as e:
                logger.warning(f"Failed to parse statistics for row: {e}")

            flattened_rows.append(flat_row)

        if flattened_rows:
            flat_df = pd.DataFrame(flattened_rows)
            # Fill missing values with 0 for numeric stat columns
            stat_cols = [col for col in flat_df.columns if col not in id_cols]
            for col in stat_cols:
                flat_df[col] = pd.to_numeric(flat_df[col], errors='coerce')
                flat_df[col] = flat_df[col].fillna(0)

            flat_df.to_csv(output_file, index=False)
            logger.info(f"✅ Created {output_file.name} with {len(flat_df)} rows and {len(flat_df.columns)} columns")
        else:
            logger.warning(f"No data to flatten from {input_file.name}")

    except Exception as e:
        logger.error(f"Failed to flatten {input_file.name}: {e}")

def post_process_match_stats() -> None:
    """Generate flattened versions of match statistics files."""
    logger.info("🔄 Post-processing match statistics...")

    # Flatten team match statistics
    flatten_stats_to_columns(
        input_file=TABLES["team_stats_match"],
        output_file=TABLES["team_stats_match_flat"],
        id_cols=['matchId', 'squadId', 'id', 'code', 'name'],
        stats_col='statistics'
    )

    # Flatten player match statistics
    flatten_stats_to_columns(
        input_file=TABLES["player_stats_match"],
        output_file=TABLES["player_stats_match_flat"],
        id_cols=['matchId', 'personId', 'id', 'squadId', 'firstName', 'lastName', 'displayName'],
        stats_col='statistics'
    )

def post_process_season_stats() -> None:
    """Generate flattened versions of season statistics and expand standings."""
    logger.info("🔄 Post-processing season statistics...")

    # Flatten team season statistics
    flatten_stats_to_columns(
        input_file=TABLES["team_stats_season"],
        output_file=OUT / "team_stats_season_flat.csv",
        id_cols=['seasonId', 'squadId', 'id', 'code', 'name'],
        stats_col='statistics'
    )

    # Flatten player season statistics (if file exists)
    if TABLES["player_stats_season"].exists():
        flatten_stats_to_columns(
            input_file=TABLES["player_stats_season"],
            output_file=OUT / "player_stats_season_flat.csv",
            id_cols=['seasonId', 'personId', 'id', 'squadId', 'firstName', 'lastName', 'displayName'],
            stats_col='statistics'
        )
    else:
        logger.warning("player_stats_season.csv not found - skipping flattening")

    # Expand standings with additional metadata
    expand_standings()

    # Flatten career statistics
    flatten_career_statistics()

def expand_standings() -> None:
    """Expand standings data with additional derived metrics."""
    standings_file = TABLES["standings"]
    if not standings_file.exists():
        logger.warning("standings.csv not found - skipping expansion")
        return

    try:
        logger.info("📈 Expanding standings data...")
        df = pd.read_csv(standings_file)

        if df.empty:
            logger.warning("standings.csv is empty")
            return

        # Add derived metrics
        if 'wins' in df.columns and 'losses' in df.columns:
            df['games_played'] = df.get('wins', 0) + df.get('losses', 0) + df.get('ties', 0)
            df['win_percentage'] = df['wins'] / df['games_played'].replace(0, 1)  # Avoid division by zero
            df['points_per_game'] = df.get('points', 0) / df['games_played'].replace(0, 1)

        # Save expanded version
        output_file = OUT / "standings_expanded.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Created {output_file.name} with {len(df)} rows and {len(df.columns)} columns")

    except Exception as e:
        logger.error(f"Failed to expand standings: {e}")

def flatten_career_statistics() -> None:
    """Flatten career statistics for both REG and POST phases."""
    phases = ["REG", "POST"]

    for phase in phases:
        career_file = TABLES[f"career_players_{phase}"]
        if not career_file.exists():
            logger.warning(f"career_players_{phase}.csv not found - skipping flattening")
            continue

        try:
            logger.info(f"📈 Flattening career statistics for {phase} phase...")
            flatten_stats_to_columns(
                input_file=career_file,
                output_file=OUT / f"career_players_{phase}_flat.csv",
                id_cols=['phase', 'personId', 'fullname', 'displayName', 'gamesPlayed'],
                stats_col='statistics'
            )
        except Exception as e:
            logger.error(f"Failed to flatten career statistics for {phase}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    post_process_match_stats()
    post_process_season_stats()
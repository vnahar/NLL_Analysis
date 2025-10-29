# Phase 1: Data Preparation & Feature Engineering

**Status:** In Progress
**Date:** 2025-10-29

## 1. Data Loading - COMPLETE ✓

### Data Sources Loaded
Successfully loaded 10 sheets from NLL_Analytics_COMPLETE.xlsx:

| Sheet Name | Rows | Columns | Key Info |
|------------|------|---------|----------|
| Schedule | 567 | 41 | All matches 2021-2025 |
| Scores Match | 567 | 24 | Match outcomes |
| Scores Period | 3,528 | 14 | Period-by-period (429 matches) |
| Team Stats Match Flat | 858 | 20 | Team stats per match (429 matches) |
| Team Stats Match Constructed | 858 | 20 | Alternative team stats |
| Team Stats Season Flat | 14 | 196 | Comprehensive season stats |
| Standings Flat | 423 | 19 | Weekly standings |
| Player Stats Match Flat | 13,948 | 9 | Player performance (513 players) |
| Player Stats Season Flat | 100 | 182 | Top player season stats |
| Teams | 58 | 6 | Team reference data |

### Unified Match Dataset
- **Total matches:** 567
- **Complete scoring data:** 567 (100%)
- **Matches with detailed stats:** 429 (75.7%)

### Data Files Created
1. `raw_data.json` - All raw data from Excel (7.16 MB)
2. `processed_matches.json` - Unified match records with targets
3. `team_stats_by_match.json` - Team stats indexed by match (429 matches)
4. `standings_lookup.json` - Standings by season/week/team (71 records)

## 2. Target Variables Created

For each match, we have three prediction targets:

### Moneyline (Match Winner)
- **Variable:** `home_win` (1 = home win, 0 = away win)
- **Baseline accuracy:** ~54.7% (home advantage)

### Point Spread
- **Variable:** `spread` (home_score - away_score)
- **Range:** Positive = home wins by X, Negative = away wins by X
- **Average:** ~0.36 goals (very competitive league)

### Total Points (Over/Under)
- **Variable:** `total` (home_score + away_score)
- **Average:** ~22.8 goals per game
- **Typical range:** 18-28 goals

## 3. Feature Engineering - IN PROGRESS

### Feature Categories to Build

#### A. Team Strength Features (Rolling Averages)
Need to calculate for each team going into a match:
- **Last 3 games:** Recent form
- **Last 5 games:** Short-term trend
- **Last 10 games:** Medium-term strength
- **Season-to-date:** Overall performance

Metrics to track:
- Goals scored per game
- Goals allowed per game
- Goal differential
- Shots per game
- Shooting percentage
- Save percentage
- Shots on goal faced
- Special teams metrics (if available)

#### B. Home/Away Splits
For each team:
- Home record (W-L, goals for/against)
- Away record (W-L, goals for/against)
- Home vs away performance differential

#### C. Context Features
- Days of rest (calculate from match dates)
- Back-to-back indicator (game on consecutive days)
- Season progress (early: weeks 1-6, mid: 7-14, late: 15+)
- Current standings position
- Win percentage at time of match

#### D. Matchup Features
- Head-to-head historical record
- Recent meetings (last 3 games vs this opponent)
- Offensive strength vs defensive weakness matchups
- Style matchups (high-scoring vs defensive teams)

#### E. Advanced Features
- Win/loss streak (current momentum)
- Scoring by period trends
- Home court advantage by venue
- Travel distance (if calculable)

## 4. Data Quality Observations

### Strengths
- Complete scoring data for all 567 matches
- No missing target variables
- Good coverage of team stats (429/567 matches)
- Multiple seasons for historical context

### Limitations
- Only 429 matches have detailed team stats (75.7%)
- Standings data: only 71 records (need to verify coverage)
- Missing some contextual data (injuries, weather, etc.)
- Limited historical depth (2021-2025, ~4 seasons)

### Data Integrity Checks Needed
1. Verify all match IDs align across sheets
2. Check for duplicate records
3. Validate date ranges and season boundaries
4. Confirm team ID consistency
5. Handle any null/missing values in key fields

## 5. Next Steps

### Immediate Tasks
1. **Build feature engineering pipeline**
   - Create rolling statistics calculator
   - Implement date-based filtering (no look-ahead bias)
   - Calculate derived features (rest days, streaks, etc.)

2. **Create training dataset**
   - Merge matches with features
   - Ensure temporal ordering (no future data leakage)
   - Split into train/validation/test sets (80/10/10)
   - Maintain chronological order in splits

3. **Feature validation**
   - Check for missing values by feature
   - Analyze feature distributions
   - Identify correlations
   - Remove redundant features

4. **Save feature set**
   - Export as `features.json`
   - Document feature definitions
   - Create data dictionary

### Timeline
- Feature engineering: Current priority
- Feature validation: Next
- Dataset preparation: After validation
- Document findings: Ongoing

## 6. Technical Notes

### Column Name Mappings
Important column name corrections made:
- Schedule: Uses `id` (not `match_id`), `squads_home_id`, `squads_away_id`, `squads_home_score_score`, etc.
- Team Stats: Uses `id` for team ID, `match_id` for match reference
- Standings: Uses `team_id`, `season_id`, `week_number`

### JSON Data Structure
All data stored as JSON arrays of objects for easy manipulation without pandas/sklearn:
```json
{
  "all_matches": [...],
  "complete_matches": [...],
  "summary": {...}
}
```

---

**Last Updated:** 2025-10-29
**Next Phase:** Feature Engineering Pipeline Implementation

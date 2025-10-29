# Phase 2: Exploratory Analysis & Feature Importance

**Status:** Complete ✓
**Date:** 2025-10-29

## Key Findings Summary

### Dataset
- **559 matches** with complete rolling statistics (98.6% of total)
- **15 teams** analyzed across seasons
- **62 features** engineered

---

## 1. Target Variable Analysis

### Moneyline (Home Win Prediction)
- **Home wins:** 305 (54.6%)
- **Away wins:** 254 (45.4%)
- **Baseline:** 54.6% accuracy by always picking home team
- **Key insight:** Moderate home advantage, not overwhelming

### Point Spread
- **Mean spread:** +0.36 goals (home favored)
- **Median:** +1.00 goals
- **Standard deviation:** 4.65 goals
- **Range:** -13 to +16 goals

**Spread Distribution:**
- Close games (±1-2 goals): 43.1% of matches
- Medium spreads (3-5 goals): 32.8% of matches
- Blowouts (6+ goals): 24.2% of matches

**Insight:** Very competitive league with many close games

### Total Points (Over/Under)
- **Mean total:** 22.78 goals
- **Median:** 23.00 goals
- **Standard deviation:** 4.62 goals
- **Range:** 9 to 37 goals

**Total Distribution:**
- Under 20 goals: 24.9%
- 20-22 goals: 25.0%
- 23-25 goals: 22.9%
- 26-28 goals: 17.5%
- 29+ goals: 9.7%

**Insight:** Scoring is relatively consistent with tight distribution around 23 goals

---

## 2. Feature Importance Rankings

### TOP 15 FEATURES FOR MONEYLINE

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | home_last10_avg_goal_diff | +0.203 | Recent goal differential strongest predictor |
| 2 | home_last3_avg_goal_diff | +0.179 | Very recent form also important |
| 3 | home_last10_avg_goals_against | -0.171 | Defensive strength matters |
| 4 | home_last10_win_pct | +0.171 | Winning percentage reliable |
| 5 | h2h_team2_wins | -0.164 | Head-to-head history significant |
| 6 | h2h_team1_win_pct | +0.163 | H2H win rate predictive |
| 7 | home_last3_wins | +0.162 | Recent wins matter |
| 8 | home_last5_avg_goal_diff | +0.162 | Medium-term form |
| 9 | home_last3_win_pct | +0.158 | Short-term win rate |
| 10 | home_last5_avg_goals_against | -0.157 | Defensive consistency |

**Key Insights:**
- **Goal differential** is the #1 predictor across all time windows
- **Last 10 games** provides best signal
- **Defensive stats** (goals against) almost as important as offensive
- **Head-to-head history** matters significantly
- **Home field advantage** shows up in features

### TOP 15 FEATURES FOR POINT SPREAD

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | home_last10_avg_goal_diff | +0.232 | Even stronger for spread prediction |
| 2 | h2h_team1_win_pct | +0.210 | H2H very important for margins |
| 3 | home_last10_win_pct | +0.210 | Win% translates to margin |
| 4 | home_last10_avg_goals_against | -0.206 | Defense critical for spreads |
| 5 | home_last3_avg_goal_diff | +0.197 | Recent form predictive |
| 6 | home_last10_wins | +0.196 | Raw wins count |
| 7 | home_last3_wins | +0.194 | Short-term wins |
| 8 | h2h_team2_wins | -0.192 | Opponent H2H record |
| 9 | home_last5_avg_goal_diff | +0.187 | Mid-term differential |
| 10 | home_last3_win_pct | +0.185 | Recent win rate |

**Key Insights:**
- **Similar to moneyline** but slightly stronger correlations
- **Goal differential** dominates (0.232 correlation)
- **Head-to-head** even more important for spread
- **Defensive metrics** highly correlated with margin control

### TOP 15 FEATURES FOR TOTAL POINTS

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | away_last10_losses | +0.119 | Weak teams allow more goals |
| 2 | away_last10_win_pct | -0.117 | Poor teams = higher scoring |
| 3 | home_last3_games_played | +0.095 | More games = better rhythm |
| 4 | week_number | -0.093 | Later season = lower scoring |
| 5 | home_last10_losses | +0.092 | Bad teams = high totals |
| 6 | away_last10_avg_goals_against | +0.090 | Poor defense = more goals |
| 7 | away_rest_days | +0.090 | Rest affects tempo |
| 8 | away_last10_avg_goal_diff | -0.089 | Good teams tighter defense |
| 9 | away_last10_wins | -0.088 | Winners control pace |
| 10 | home_last5_games_played | +0.083 | Game count affects scoring |

**Key Insights:**
- **Weaker correlations** overall (total points harder to predict)
- **Team quality paradox:** Bad teams = higher totals (poor defense)
- **Season timing matters:** Later games score less
- **Rest impacts pace:** More rest = different tempo
- **Both teams' defense matters:** Need combined analysis

---

## 3. Context Effects Analysis

### Season Phase Effects

| Phase | Games | Home Win % | Avg Total | Avg |Spread| |
|-------|-------|------------|-----------|--------------|
| **Early** (Weeks 1-6) | 111 | 45.0% | 23.21 | 4.02 |
| **Mid** (Weeks 7-14) | 208 | 54.3% | 23.09 | 3.50 |
| **Late** (Weeks 15+) | 240 | 59.2% | 22.30 | 3.84 |

**Insights:**
- **Home advantage increases** as season progresses (45% → 59%)
- **Scoring decreases** slightly in late season (23.2 → 22.3 goals)
- **Closer games mid-season** (3.50 avg spread)
- **Early season unpredictability** - home teams struggle

### Back-to-Back Game Effects

| Situation | Games | Win % |
|-----------|-------|-------|
| Home team on B2B | 20 | 60.0% |
| Home team with rest | 529 | 54.4% |
| Away team on B2B | 19 | 26.3% |
| Away team with rest | 532 | 46.1% |

**Insights:**
- **Massive B2B effect for away teams** (46.1% → 26.3% win rate)
- **Home teams actually benefit from B2B** (54.4% → 60.0%)
- **Home court advantage amplified** when opponent on B2B
- **Critical feature** for prediction models

### Win Streak Effects

| Streak Status | Instances | Win % |
|---------------|-----------|-------|
| Hot (3+ wins) | 160 | 58.8% |
| Warm (1-2 wins) | 405 | 47.9% |
| Cold (-1 to -2 losses) | 406 | 49.8% |
| Ice Cold (3+ losses) | 147 | 46.9% |

**Insights:**
- **Hot teams maintain momentum** (58.8% vs baseline 50%)
- **Modest streak effect** (+8-10% for hot teams)
- **Losing streaks less predictive** (cold teams still ~47-50%)
- **Momentum matters but not overwhelming**

---

## 4. Team Performance Profiles

### Elite Teams (Top 3)
1. **Team 509:** 74.5% win rate (78.0% home, 70.5% away) - Dominant
2. **Team 515:** 61.3% win rate (63.4% home, 59.0% away) - Strong road team
3. **Team 542:** 61.3% win rate (68.3% home, 53.8% away) - Home-dependent

### Weak Teams (Bottom 3)
1. **Team 549:** 26.4% win rate (29.6% home, 23.1% away) - Consistently poor
2. **Team 545:** 38.0% win rate (40.0% home, 36.1% away) - Below average
3. **Team 543:** 39.2% win rate (47.4% home, 30.6% away) - Terrible road record

### Key Observations
- **Wide talent disparity:** 74.5% to 26.4% win rates (48% spread)
- **Home/away splits vary:** Some teams road warriors, others home-dependent
- **Team 509 dominance:** Far above competition
- **Predictability:** Elite vs weak matchups highly predictable

---

## 5. Predictive Insights for Modeling

### Moneyline Predictions
**Best predictors:**
1. Last 10 game goal differential (home team)
2. Head-to-head win percentage
3. Last 10 defensive performance
4. Recent win percentage (last 3-5 games)
5. Away team on back-to-back

**Expected accuracy:** 60-65% (beating 54.6% baseline)

### Spread Predictions
**Best predictors:**
1. Last 10 goal differential (both teams)
2. Head-to-head history
3. Home win percentage
4. Defensive metrics (goals against)
5. Season phase (early vs late)

**Expected MAE:** 2.5-3.5 goals

### Total Predictions
**Best predictors:**
1. Away team quality (win%, losses)
2. Combined defensive ratings
3. Week number (season phase)
4. Rest days (both teams)
5. Team offensive averages

**Expected MAE:** 3.0-4.0 goals
**Challenge:** Weakest correlations of three bet types

---

## 6. Model Development Recommendations

### Feature Selection
**Must-include features (Top tier):**
- `home_last10_avg_goal_diff`
- `away_last10_avg_goal_diff`
- `h2h_team1_win_pct`
- `home_last10_avg_goals_against`
- `away_last10_avg_goals_against`
- `home/away_back_to_back` indicators

**Important features (Second tier):**
- Last 3/5 game rolling stats
- Home/away split performance
- Streak indicators
- Season phase
- Rest days

**Lower priority:**
- Week number (already captured in phase)
- Games played (correlated with other features)
- Individual game counts

### Model Approaches
1. **Logistic Regression (Moneyline):** Should perform well with linear relationships
2. **Weighted Scoring Model:** Combine top features with correlation weights
3. **Ensemble:** Combine multiple simple models
4. **Separate Models:** Consider season phase-specific models

### Data Split Strategy
- **Temporal split:** Train on earlier seasons, test on later
- **Walk-forward validation:** Respect chronological order
- **80/10/10 split:** Train/Validation/Test

---

## 7. Key Challenges Identified

1. **Total points hardest to predict** (weak correlations ~0.12 max)
2. **Early season unpredictability** (only 45% home win rate)
3. **Small sample on B2B games** (only 19-20 instances)
4. **Team quality changes** over seasons (need recency weighting)
5. **Close game frequency** (43% within 2 goals - hard to predict spread)

---

**Next Steps:** Build prediction models using identified top features

**Last Updated:** 2025-10-29

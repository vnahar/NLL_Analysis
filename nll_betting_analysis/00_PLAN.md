# NLL Betting Prediction Model - Master Plan

**Objective:** Build predictive models for Moneyline, Point Spread, and Total Points using NLL_Analytics_COMPLETE.xlsx

**Date Started:** 2025-10-29

## Project Structure
```
nll_betting_analysis/
├── 00_PLAN.md (this file)
├── 01_data_preparation.md
├── 02_exploratory_analysis.md
├── 03_model_development.md
├── 04_feature_importance.md
├── 05_final_report.md
├── data/
│   ├── raw_data.json
│   ├── processed_matches.json
│   ├── features.json
│   └── train_test_split.json
├── models/
│   ├── moneyline_model.json
│   ├── spread_model.json
│   ├── total_model.json
│   └── predictions.json
└── scripts/
    ├── data_loader.py
    ├── feature_engineering.py
    ├── models.py
    └── analysis.py
```

## Phase 1: Data Preparation & Feature Engineering

### 1.1 Core Data Integration
- Load key sheets: Schedule, Scores Match, Team Stats Match, Standings Flat
- Create unified match dataset (429 matches with complete stats)
- Create target variables:
  - **Moneyline**: Binary home win (1/0)
  - **Spread**: home_score - away_score
  - **Total**: home_score + away_score

### 1.2 Feature Engineering
**Team Strength Features:**
- Rolling averages (3, 5, 10 games): goals scored/allowed, shots, save %, shooting %
- Season-to-date cumulative stats
- Home/away performance splits
- Special teams ratings (PP%, PK%)

**Context Features:**
- Home advantage indicator (baseline: 54.7% home win rate)
- Days rest between games
- Back-to-back game indicator
- Season progress (early/mid/late)
- Current standings position and win %

**Matchup Features:**
- Head-to-head historical results
- Offensive vs defensive matchup
- Goal differential trends
- Shot differential

**Advanced Features:**
- Goalie performance metrics
- Momentum indicators (win/loss streaks)
- Scoring by period patterns
- Key player performance

### 1.3 Data Quality Handling
- Handle missing venue data (24%)
- Handle matches without detailed stats (138 matches)
- Create train/validation/test split (temporal ordering)

## Phase 2: Exploratory Analysis & Feature Importance

### 2.1 Univariate Analysis
- Distribution of outcomes (win rates, spreads, totals)
- Team-level performance variations
- Home vs away differences
- Seasonal trends

### 2.2 Feature Correlation Analysis
For each bet type:
- **Moneyline**: Features most correlated with wins
- **Spread**: Margin prediction indicators
- **Totals**: Pace/efficiency drivers

### 2.3 Key Insights Extraction
- Team over/underperformance
- Home advantage by venue
- High vs low scoring patterns
- Competitive balance analysis

## Phase 3: Model Development

### 3.1 Baseline Models
- **Moneyline**: Home team wins (54.7% baseline)
- **Spread**: Historical average margin
- **Totals**: Sum of team scoring averages

### 3.2 Custom ML Models (No sklearn)
Implement from scratch using JSON objects:
1. **Logistic Regression** (manual gradient descent)
2. **Weighted Scoring System** (feature-based)
3. **Simple Ensemble** (combine multiple approaches)

### 3.3 Model Optimization
- Manual hyperparameter tuning
- Feature selection (correlation-based)
- Cross-validation (time-series aware)

### 3.4 Evaluation Metrics
- **Moneyline**: Accuracy, ROI simulation
- **Spread**: MAE, RMSE, directional accuracy
- **Totals**: MAE, RMSE, over/under accuracy

## Phase 4: Feature Importance & Insights

### 4.1 Feature Importance
- Correlation coefficients
- Manual permutation testing
- Rank top 20 features per bet type

### 4.2 Key Findings
- What drives wins?
- Spread predictability factors
- Total scoring drivers
- Temporal patterns
- Team archetypes

### 4.3 Practical Recommendations
- Confidence intervals
- Most predictable bet types
- Edge identification
- High uncertainty situations

## Phase 5: Deliverables

### 5.1 Analysis Documentation
- Complete markdown reports per phase
- Visualizations (ASCII charts in markdown)
- Model performance comparisons
- Feature importance tables

### 5.2 Prediction System
- Trained models as JSON
- Prediction functions
- Probability outputs for moneyline
- Point estimates + confidence for spread/totals

### 5.3 Insights Report
- Executive summary
- Top predictive features
- Model accuracy and limitations
- Betting strategy recommendations

## Technical Approach

**Implementation:**
- Python with pandas, openpyxl for Excel reading
- JSON for data storage (no sklearn/ml libraries)
- Custom implementations of prediction algorithms
- Markdown for documentation

**Validation Strategy:**
- Time-series split (respect temporal ordering)
- Walk-forward validation
- Manual cross-validation

**Expected Challenges:**
- Small dataset (429 matches)
- Very competitive league (0.36 goal avg spread)
- Limited historical depth (2021-2025)
- Missing some event-level data

**Success Criteria:**
- Moneyline: >60% accuracy (beat 54.7% baseline)
- Spread: <2.5 goals MAE
- Totals: <3 goals MAE

## Status Tracking

- [ ] Phase 1: Data Preparation
- [ ] Phase 2: Exploratory Analysis
- [ ] Phase 3: Model Development
- [ ] Phase 4: Feature Importance
- [ ] Phase 5: Final Report

---

**Last Updated:** 2025-10-29

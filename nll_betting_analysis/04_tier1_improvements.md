# Tier 1 Feature Improvements - Results

**Date:** 2025-10-29
**Status:** Completed - Mixed Results

## Objective
Implement 8 high-impact, low-complexity features to improve model accuracy through:
- Interaction terms (offense × defense)
- Ratio features (relative strength)
- Differential features (asymmetric advantages)
- Decay-weighted features (recency emphasis)

---

## Features Added (8 Total)

### 1. Interaction Features
- `interaction_home_off_away_def`: home_goals_for × away_goals_against / 100
- `interaction_away_off_home_def`: away_goals_for × home_goals_against / 100
- **Purpose:** Capture how strong offense performs vs specific defense quality

### 2. Ratio Features
- `team_quality_ratio`: home_win_pct / away_win_pct
- `goal_diff_ratio`: home_goal_diff / away_goal_diff
- **Purpose:** Relative strength comparison (ratio better than absolute difference)

### 3. Differential Features
- `rest_advantage`: home_rest_days - away_rest_days
- `b2b_differential`: away_back_to_back - home_back_to_back
- **Purpose:** Capture asymmetric fatigue advantages

### 4. Decay-Weighted Features
- `home_weighted_goal_diff`: (last3 × 2 + last10) / 3
- `away_weighted_goal_diff`: (last3 × 2 + last10) / 3
- **Purpose:** Recent games weighted more heavily (recency bias)

---

## Implementation Details

### Feature Engineering
- Updated [feature_engineering.py](scripts/feature_engineering.py:313-375)
- Total features: 62 → **70** (+8 new features)
- Feature completeness: 98.6% (560/567 matches)
- File size: 1.65 MB → 1.83 MB

### Model Updates
- Moneyline: 10 → 9 features (replaced redundant ones)
- Spread: 9 → 9 features (swapped old for new)
- Total: 9 → 9 features (added interactions)

---

## Results Comparison

### MONEYLINE (Match Winner)

| Model | Original | Tier 1 | Change | Status |
|-------|----------|--------|--------|--------|
| **Baseline** | 50.00% | 50.00% | 0.00% | - |
| **Weighted** | 48.81% | 48.81% | 0.00% | Same |
| **Logistic** | **58.33%** | **55.95%** | **-2.38%** | ❌ Worse |

**Analysis:**
- Moneyline accuracy **decreased** by 2.38 percentage points
- Likely due to:
  1. **Multicollinearity**: New weighted features highly correlated with old (r=0.829)
  2. **Information loss**: Removed some predictive features (win_pct, individual b2b)
  3. **Model capacity**: Logistic regression may not leverage ratios effectively

### POINT SPREAD (Margin Prediction)

| Model | Original MAE | Tier 1 MAE | Change | Status |
|-------|--------------|------------|--------|--------|
| **Baseline** | 3.96 | 3.96 | 0.00 | - |
| **Weighted** | 6.53 | 6.47 | **-0.06** | ✅ Tiny improvement |

**Analysis:**
- Essentially **no change** (0.06 goals improvement)
- Still worse than baseline (6.47 vs 3.96)
- Weighted model continues to underperform

### TOTAL POINTS (Over/Under)

| Model | Original MAE | Tier 1 MAE | Change | Status |
|-------|--------------|------------|--------|--------|
| **Baseline** | 3.76 | 3.76 | 0.00 | - |
| **Weighted** | 4.56 | 4.01 | **-0.55** | ✅ Improvement |

**Analysis:**
- Total MAE **improved** by 0.55 goals (12% better)
- **Interaction features** helping! (offense × defense)
- Still worse than baseline, but moving in right direction
- This was expected - interactions are key for totals

---

## Key Findings

### ✅ What Worked

**1. Interaction Features (for Totals)**
- `interaction_home_off_away_def` and `interaction_away_off_home_def` helped totals prediction
- MAE dropped from 4.56 → 4.01 goals
- **Validates hypothesis**: Totals need offense × defense interactions

**2. Feature Engineering Process**
- Successfully added 8 new features
- All features have >98% completeness
- No technical issues in implementation

### ❌ What Didn't Work

**1. Weighted Goal Differential**
- High correlation with original (r=0.829)
- Caused multicollinearity issues
- Didn't add new information, just reweighted existing

**2. Ratio Features**
- `team_quality_ratio` and `goal_diff_ratio` didn't help moneyline
- May need non-linear models to leverage properly
- Linear models struggle with ratios

**3. Feature Replacement Strategy**
- Removing old features and replacing with new ones backfired
- Lost predictive power from original features
- Should have **added** features, not replaced

---

## Analysis: Why Moneyline Got Worse

### Problem 1: Multicollinearity
```
home_last10_avg_goal_diff  vs  home_weighted_goal_diff
Correlation: r = 0.829 (very high!)

This means they're measuring nearly the same thing.
Model gets confused when two features tell the same story.
```

###Problem 2: Information Loss
**Removed:**
- `home_last10_win_pct` (r=0.171 with target)
- `away_last10_win_pct` (r=-0.171 with target)
- `home_back_to_back` (important signal)
- `away_back_to_back` (r=0.20 with target!)

**Replaced with:**
- `team_quality_ratio` (less interpretable)
- `b2b_differential` (aggregated signal)

**Result:** Lost granular information that logistic regression was using

### Problem 3: Model Type Mismatch
- **Linear models** (logistic regression) work best with additive features
- **Ratio features** benefit non-linear models (trees, neural nets)
- We gave linear model ratio features it can't leverage well

---

## Lessons Learned

### 1. Don't Replace, Add
❌ **Wrong approach:**
```python
# Remove old features, add new ones
features = ['old_feature_A', 'old_feature_B']
features = ['new_feature_A', 'new_feature_B']  # Replaced!
```

✅ **Right approach:**
```python
# Keep old features, add new ones
features = ['old_feature_A', 'old_feature_B',
            'new_feature_A', 'new_feature_B']  # Added!
```

Then use feature selection (L1 regularization, LASSO) to remove redundant ones automatically.

### 2. Check Multicollinearity First
Before adding features, check correlation:
- If r > 0.7: High multicollinearity, reconsider
- If r < 0.5: Low correlation, safe to add
- Our weighted features had r=0.829 (too high!)

### 3. Match Features to Model Type
- **Linear models:** Additive features (differences, sums)
- **Tree models:** Ratios, interactions work great
- **Neural nets:** Can learn interactions automatically

We tried to force ratios into linear model = bad fit.

### 4. Interactions Work for Totals!
The one success: **interaction features for totals**
- Offense × Defense captures what linear sum cannot
- 12% improvement (4.56 → 4.01 MAE)
- Should expand on this for Tier 2

---

## Recommendations Going Forward

### Short-term: Revert Moneyline Changes
✅ **Keep original 10 moneyline features** (58.33% accuracy)
✅ **Keep interaction features for totals** (4.01 MAE is better)
❌ Don't use ratio/weighted features with logistic regression

### Medium-term: Try Tree-Based Models
- XGBoost or LightGBM can leverage ratios and interactions
- These models handle multicollinearity better
- Can automatically learn feature interactions
- Expected: 60-65% moneyline accuracy

### Long-term: Feature Selection Pipeline
1. Add ALL features (original + new = 70 total)
2. Use L1 regularization (LASSO) to automatically remove redundant ones
3. Let model decide which features are useful
4. This avoids manual feature replacement mistakes

---

## Tier 2 Plan (Revised)

Based on learnings:

### 1. Mine Existing Data
- **Goalie stats** from Player Stats Season sheet (SV%, W-L)
- **Shot metrics** from Shots data (accuracy, volume)
- **Special teams** from Team Stats (PP%, PK%)

### 2. Build Tree-Based Models
- XGBoost for moneyline (can use ratio features)
- XGBoost for spread (handles interactions)
- XGBoost for totals (leverage offense × defense)

### 3. Ensemble Approach
- Combine logistic regression (58.33%) + XGBoost
- Weighted average or stacking
- Expected: 60-65% accuracy

### 4. Feature Selection
- Use LASSO or feature importance from trees
- Remove truly redundant features
- Keep diversity of information

---

## Updated Performance Targets

| Bet Type | Current Best | Tier 1 Result | Revised Target |
|----------|--------------|---------------|----------------|
| **Moneyline** | 58.33% (original) | 55.95% (tier 1) | Keep original, aim for 60-62% with XGBoost |
| **Spread** | 3.96 MAE (baseline) | 6.47 MAE (tier 1) | 3.2-3.5 MAE with XGBoost |
| **Total** | 3.76 MAE (baseline) | 4.01 MAE (tier 1) | 3.0-3.2 MAE with interactions + XGBoost |

---

## Code Changes Made

### Files Modified:
1. [feature_engineering.py](scripts/feature_engineering.py) - Added 8 new features ✅
2. [models.py](scripts/models.py) - Updated feature lists ✅
3. [features.json](data/features.json) - Regenerated with 70 features ✅

### Files to Revert (for production):
- Revert `models.py` moneyline features to original 10
- Keep total features with interactions
- Keep spread features (no harm, no benefit)

---

## Conclusion

### Summary
- **Added 8 Tier 1 features successfully** ✅
- **Interaction features helped totals** ✅ (12% improvement)
- **Ratio/weighted features hurt moneyline** ❌ (-2.38% accuracy)
- **Root cause:** Multicollinearity + feature replacement strategy

### Key Takeaway
> "Adding more features isn't always better. The right features for the right model type, with proper selection, is what matters."

### Next Steps
1. ✅ Keep interaction features for totals
2. ❌ Revert moneyline to original features (58.33%)
3. 🔨 Build XGBoost models in Tier 2
4. 🔨 Mine goalie and shot data
5. 🔨 Implement feature selection pipeline

---

**Status:** Tier 1 partially successful - learned valuable lessons for Tier 2

**Last Updated:** 2025-10-29

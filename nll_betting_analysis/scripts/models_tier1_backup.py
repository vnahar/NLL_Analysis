"""
NLL Betting Analysis - Prediction Models
Custom implementations without sklearn
"""

import json
from pathlib import Path
import statistics
import math
import random

class DataSplitter:
    """Handle train/test splits with temporal ordering"""

    @staticmethod
    def temporal_split(data, train_ratio=0.8, val_ratio=0.1):
        """Split data maintaining chronological order"""
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            'train': data[:train_end],
            'val': data[train_end:val_end],
            'test': data[val_end:]
        }


class BaselineModel:
    """Baseline predictions for comparison"""

    def predict_moneyline_always_home(self, test_data):
        """Always predict home win"""
        predictions = [1 for _ in test_data]
        actuals = [d['home_win'] for d in test_data]
        accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(actuals)
        return {'predictions': predictions, 'accuracy': accuracy}

    def predict_spread_historical_avg(self, train_data, test_data):
        """Predict using historical average spread"""
        avg_spread = statistics.mean([d['spread'] for d in train_data if d['spread'] is not None])

        predictions = [avg_spread for _ in test_data]
        actuals = [d['spread'] for d in test_data]

        mae = statistics.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        rmse = math.sqrt(statistics.mean([(p - a)**2 for p, a in zip(predictions, actuals)]))

        return {'predictions': predictions, 'mae': mae, 'rmse': rmse, 'avg_spread': avg_spread}

    def predict_total_sum_of_avgs(self, train_data, test_data):
        """Predict using average total"""
        avg_total = statistics.mean([d['total'] for d in train_data if d['total'] is not None])

        predictions = [avg_total for _ in test_data]
        actuals = [d['total'] for d in test_data]

        mae = statistics.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        rmse = math.sqrt(statistics.mean([(p - a)**2 for p, a in zip(predictions, actuals)]))

        return {'predictions': predictions, 'mae': mae, 'rmse': rmse, 'avg_total': avg_total}


class WeightedScoringModel:
    """Weighted feature scoring model based on correlations"""

    def __init__(self):
        self.weights = {}
        self.threshold = 0.5

    def train_moneyline(self, train_data, features):
        """Train using feature correlations"""
        print("  Training weighted scoring model for moneyline...")

        # Calculate weights from correlations
        for feature in features:
            values = [d[feature] for d in train_data if d.get(feature) is not None]
            targets = [d['home_win'] for d in train_data if d.get(feature) is not None]

            if len(values) < 10:
                continue

            # Simple correlation
            corr = self._correlation(values, targets)
            self.weights[feature] = corr

        print(f"    Learned {len(self.weights)} feature weights")

    def predict_moneyline(self, test_data, features):
        """Predict using weighted sum"""
        predictions = []

        for sample in test_data:
            score = 0
            weight_sum = 0

            for feature in features:
                if feature in self.weights and sample.get(feature) is not None:
                    # Normalize feature value
                    val = sample[feature]
                    score += self.weights[feature] * val
                    weight_sum += abs(self.weights[feature])

            # Normalize score
            if weight_sum > 0:
                score = score / weight_sum

            # Predict home win if score > threshold
            pred = 1 if score > 0 else 0
            predictions.append(pred)

        actuals = [d['home_win'] for d in test_data]
        accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(actuals)

        return {'predictions': predictions, 'accuracy': accuracy}

    def train_spread(self, train_data, features):
        """Train spread predictor"""
        print("  Training weighted scoring model for spread...")

        for feature in features:
            values = [d[feature] for d in train_data if d.get(feature) is not None]
            targets = [d['spread'] for d in train_data if d.get(feature) is not None]

            if len(values) < 10:
                continue

            corr = self._correlation(values, targets)
            self.weights[feature] = corr

        print(f"    Learned {len(self.weights)} feature weights")

    def predict_spread(self, test_data, features):
        """Predict spread using weighted sum"""
        predictions = []

        for sample in test_data:
            score = 0
            weight_sum = 0

            for feature in features:
                if feature in self.weights and sample.get(feature) is not None:
                    val = sample[feature]
                    score += self.weights[feature] * val
                    weight_sum += abs(self.weights[feature])

            # Scale to spread range
            if weight_sum > 0:
                score = score / weight_sum * 5  # Scale factor

            predictions.append(score)

        actuals = [d['spread'] for d in test_data]
        mae = statistics.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        rmse = math.sqrt(statistics.mean([(p - a)**2 for p, a in zip(predictions, actuals)]))

        return {'predictions': predictions, 'mae': mae, 'rmse': rmse}

    def train_total(self, train_data, features):
        """Train total predictor"""
        print("  Training weighted scoring model for total...")

        # Start with average as base
        self.base_total = statistics.mean([d['total'] for d in train_data])

        for feature in features:
            values = [d[feature] for d in train_data if d.get(feature) is not None]
            targets = [d['total'] - self.base_total for d in train_data if d.get(feature) is not None]

            if len(values) < 10:
                continue

            corr = self._correlation(values, targets)
            self.weights[feature] = corr

        print(f"    Learned {len(self.weights)} feature weights, base={self.base_total:.2f}")

    def predict_total(self, test_data, features):
        """Predict total using weighted sum + base"""
        predictions = []

        for sample in test_data:
            adjustment = 0
            weight_sum = 0

            for feature in features:
                if feature in self.weights and sample.get(feature) is not None:
                    val = sample[feature]
                    adjustment += self.weights[feature] * val
                    weight_sum += abs(self.weights[feature])

            # Scale adjustment
            if weight_sum > 0:
                adjustment = adjustment / weight_sum * 2  # Scale factor

            pred = self.base_total + adjustment
            predictions.append(pred)

        actuals = [d['total'] for d in test_data]
        mae = statistics.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        rmse = math.sqrt(statistics.mean([(p - a)**2 for p, a in zip(predictions, actuals)]))

        return {'predictions': predictions, 'mae': mae, 'rmse': rmse}

    def _correlation(self, x, y):
        """Calculate correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x)**2 for xi in x)**0.5
        denom_y = sum((yi - mean_y)**2 for yi in y)**0.5

        if denom_x == 0 or denom_y == 0:
            return 0

        return numerator / (denom_x * denom_y)


class SimpleLogisticRegression:
    """Simple logistic regression using gradient descent"""

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = {}
        self.bias = 0

    def _sigmoid(self, z):
        """Sigmoid function with numerical stability"""
        z = max(min(z, 500), -500)  # Clip to prevent overflow
        return 1 / (1 + math.exp(-z))

    def train(self, train_data, features):
        """Train using gradient descent"""
        print(f"  Training logistic regression for {self.iterations} iterations...")

        # Initialize weights
        for feature in features:
            self.weights[feature] = random.uniform(-0.1, 0.1)

        # Filter complete data
        complete_data = [d for d in train_data if all(d.get(f) is not None for f in features)]
        print(f"    Training on {len(complete_data)} complete samples")

        # Gradient descent
        for iteration in range(self.iterations):
            # Calculate gradients
            grad_weights = {f: 0 for f in features}
            grad_bias = 0

            for sample in complete_data:
                # Forward pass
                z = self.bias
                for feature in features:
                    z += self.weights[feature] * sample[feature]

                pred = self._sigmoid(z)
                error = pred - sample['home_win']

                # Backward pass
                grad_bias += error
                for feature in features:
                    grad_weights[feature] += error * sample[feature]

            # Update weights
            n = len(complete_data)
            self.bias -= self.lr * grad_bias / n
            for feature in features:
                self.weights[feature] -= self.lr * grad_weights[feature] / n

            if (iteration + 1) % 200 == 0:
                # Calculate loss
                loss = 0
                for sample in complete_data:
                    z = self.bias
                    for feature in features:
                        z += self.weights[feature] * sample[feature]
                    pred = self._sigmoid(z)
                    loss += -(sample['home_win'] * math.log(pred + 1e-10) +
                            (1 - sample['home_win']) * math.log(1 - pred + 1e-10))
                loss /= n
                print(f"    Iteration {iteration+1}: Loss = {loss:.4f}")

    def predict(self, test_data, features):
        """Make predictions"""
        predictions = []
        probabilities = []

        for sample in test_data:
            # Check if all features available
            if not all(sample.get(f) is not None for f in features):
                predictions.append(1)  # Default to home win
                probabilities.append(0.5)
                continue

            z = self.bias
            for feature in features:
                z += self.weights[feature] * sample[feature]

            prob = self._sigmoid(z)
            pred = 1 if prob > 0.5 else 0

            predictions.append(pred)
            probabilities.append(prob)

        actuals = [d['home_win'] for d in test_data]
        accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(actuals)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'accuracy': accuracy
        }


class ModelEvaluator:
    """Evaluate and compare models"""

    @staticmethod
    def evaluate_moneyline(predictions, actuals):
        """Calculate moneyline metrics"""
        accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(actuals)

        # Calculate by class
        home_correct = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
        home_total = sum(1 for a in actuals if a == 1)
        away_correct = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 0)
        away_total = sum(1 for a in actuals if a == 0)

        home_precision = home_correct / sum(1 for p in predictions if p == 1) if sum(predictions) > 0 else 0
        away_precision = away_correct / sum(1 for p in predictions if p == 0) if sum(1 for p in predictions if p == 0) > 0 else 0

        return {
            'accuracy': accuracy,
            'home_precision': home_precision,
            'away_precision': away_precision,
            'home_recall': home_correct / home_total if home_total > 0 else 0,
            'away_recall': away_correct / away_total if away_total > 0 else 0
        }

    @staticmethod
    def evaluate_regression(predictions, actuals):
        """Calculate regression metrics"""
        mae = statistics.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        rmse = math.sqrt(statistics.mean([(p - a)**2 for p, a in zip(predictions, actuals)]))

        # Within X goals
        within_1 = sum(1 for p, a in zip(predictions, actuals) if abs(p - a) <= 1) / len(actuals)
        within_2 = sum(1 for p, a in zip(predictions, actuals) if abs(p - a) <= 2) / len(actuals)
        within_3 = sum(1 for p, a in zip(predictions, actuals) if abs(p - a) <= 3) / len(actuals)

        return {
            'mae': mae,
            'rmse': rmse,
            'within_1': within_1,
            'within_2': within_2,
            'within_3': within_3
        }


def main():
    """Main execution"""
    print("="*60)
    print("NLL BETTING ANALYSIS - MODEL TRAINING")
    print("="*60)

    data_dir = Path('/Users/vedantnahar/Downloads/AltSportsData/NLL_Analysis/nll_betting_analysis/data')

    # Load features
    print("\nLoading features...")
    with open(data_dir / 'features.json', 'r') as f:
        data = json.load(f)
        features_data = data['features']

    # Filter complete data
    complete_data = [
        f for f in features_data
        if f.get('home_last5_avg_goals_for') is not None
        and f.get('away_last5_avg_goals_for') is not None
    ]
    print(f"  ✓ Loaded {len(complete_data)} matches with complete features")

    # Split data
    print("\nSplitting data temporally...")
    splits = DataSplitter.temporal_split(complete_data, train_ratio=0.7, val_ratio=0.15)
    print(f"  Train: {len(splits['train'])} matches")
    print(f"  Val:   {len(splits['val'])} matches")
    print(f"  Test:  {len(splits['test'])} matches")

    # Define top features from analysis + TIER 1 IMPROVEMENTS
    # NOTE: Replacing some old features with improved versions to avoid multicollinearity
    moneyline_features = [
        # Keep these original features (not replaced)
        'home_last10_avg_goals_against',
        'h2h_team1_win_pct',
        'home_streak',
        'away_streak',
        # TIER 1: Replace goal_diff with weighted version (r=0.829 correlation)
        'home_weighted_goal_diff',
        'away_weighted_goal_diff',
        # TIER 1: Replace win_pct with ratio (relative strength better)
        'team_quality_ratio',
        # TIER 1: Replace individual b2b with differential
        'b2b_differential',
        # TIER 1: Add goal_diff_ratio (new information)
        'goal_diff_ratio'
    ]

    spread_features = [
        # Keep these original features
        'h2h_team1_win_pct',
        'home_last10_avg_goals_against',
        'away_last10_avg_goals_against',
        # TIER 1: Replace goal_diff with weighted version
        'home_weighted_goal_diff',
        'away_weighted_goal_diff',
        # TIER 1: Add ratios and differentials
        'team_quality_ratio',
        'goal_diff_ratio',
        'rest_advantage',
        'b2b_differential'
    ]

    total_features = [
        # Keep base offensive/defensive stats
        'home_last10_avg_goals_for',
        'away_last10_avg_goals_for',
        'home_last10_avg_goals_against',
        'away_last10_avg_goals_against',
        'week_number',
        # TIER 1: KEY INTERACTIONS (offense × defense)
        'interaction_home_off_away_def',
        'interaction_away_off_home_def',
        # TIER 1: Add team quality and rest
        'team_quality_ratio',
        'rest_advantage'
    ]

    results = {}

    # MONEYLINE MODELS
    print("\n" + "="*60)
    print("MONEYLINE PREDICTION")
    print("="*60)

    # Baseline
    print("\n1. Baseline (Always Home)")
    baseline = BaselineModel()
    bl_results = baseline.predict_moneyline_always_home(splits['test'])
    print(f"   Accuracy: {bl_results['accuracy']*100:.2f}%")
    results['moneyline_baseline'] = bl_results

    # Weighted Scoring
    print("\n2. Weighted Scoring Model")
    ws_model = WeightedScoringModel()
    ws_model.train_moneyline(splits['train'], moneyline_features)
    ws_results = ws_model.predict_moneyline(splits['test'], moneyline_features)
    print(f"   Accuracy: {ws_results['accuracy']*100:.2f}%")
    results['moneyline_weighted'] = ws_results

    # Logistic Regression
    print("\n3. Logistic Regression")
    lr_model = SimpleLogisticRegression(learning_rate=0.001, iterations=1000)
    lr_model.train(splits['train'], moneyline_features)
    lr_results = lr_model.predict(splits['test'], moneyline_features)
    print(f"   Accuracy: {lr_results['accuracy']*100:.2f}%")
    results['moneyline_logistic'] = lr_results

    # SPREAD MODELS
    print("\n" + "="*60)
    print("POINT SPREAD PREDICTION")
    print("="*60)

    # Baseline
    print("\n1. Baseline (Historical Average)")
    spread_bl = baseline.predict_spread_historical_avg(splits['train'], splits['test'])
    print(f"   MAE: {spread_bl['mae']:.2f} goals")
    print(f"   RMSE: {spread_bl['rmse']:.2f} goals")
    results['spread_baseline'] = spread_bl

    # Weighted Scoring
    print("\n2. Weighted Scoring Model")
    ws_spread = WeightedScoringModel()
    ws_spread.train_spread(splits['train'], spread_features)
    ws_spread_results = ws_spread.predict_spread(splits['test'], spread_features)
    print(f"   MAE: {ws_spread_results['mae']:.2f} goals")
    print(f"   RMSE: {ws_spread_results['rmse']:.2f} goals")
    results['spread_weighted'] = ws_spread_results

    # TOTAL MODELS
    print("\n" + "="*60)
    print("TOTAL POINTS PREDICTION")
    print("="*60)

    # Baseline
    print("\n1. Baseline (Average Total)")
    total_bl = baseline.predict_total_sum_of_avgs(splits['train'], splits['test'])
    print(f"   MAE: {total_bl['mae']:.2f} goals")
    print(f"   RMSE: {total_bl['rmse']:.2f} goals")
    results['total_baseline'] = total_bl

    # Weighted Scoring
    print("\n2. Weighted Scoring Model")
    ws_total = WeightedScoringModel()
    ws_total.train_total(splits['train'], total_features)
    ws_total_results = ws_total.predict_total(splits['test'], total_features)
    print(f"   MAE: {ws_total_results['mae']:.2f} goals")
    print(f"   RMSE: {ws_total_results['rmse']:.2f} goals")
    results['total_weighted'] = ws_total_results

    # Save results
    output_file = data_dir / 'model_results.json'

    # Convert to serializable format
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {}
        for k, v in value.items():
            if isinstance(v, list):
                serializable_results[key][k] = v[:10]  # Save only first 10 predictions
            else:
                serializable_results[key][k] = v

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "="*60)
    print(f"✓ Results saved to {output_file}")
    print("="*60)


if __name__ == '__main__':
    main()

"""
NLL Betting Analysis - Prediction Models
Simulation-based approach using period-level negative binomial distribution

MIGRATION NOTE:
- Original logistic regression model backed up to models_logistic_backup.py
- This file now uses simulation-based prediction
- API maintained for compatibility with existing code
"""

import json
from pathlib import Path
import statistics
import math

# Import simulation model
from simulation_models import NegativeBinomialSimulator, SimulationBasedPredictor


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

    @staticmethod
    def temporal_split_by_season(data, train_seasons, test_seasons):
        """
        Split data by season IDs for stricter temporal separation

        Args:
            data: List of match records with 'season_id' field
            train_seasons: List of season IDs for training (e.g., [221, 222, 223])
            test_seasons: List of season IDs for testing (e.g., [224])

        Returns:
            Dict with 'train' and 'test' keys
        """
        train = [d for d in data if d.get('season_id') in train_seasons]
        test = [d for d in data if d.get('season_id') in test_seasons]

        return {
            'train': train,
            'test': test
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

    def predict_total_historical_avg(self, train_data, test_data):
        """Predict using historical average total"""
        avg_total = statistics.mean([d['total'] for d in train_data if d['total'] is not None])

        predictions = [avg_total for _ in test_data]
        actuals = [d['total'] for d in test_data]

        mae = statistics.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        rmse = math.sqrt(statistics.mean([(p - a)**2 for p, a in zip(predictions, actuals)]))

        return {'predictions': predictions, 'mae': mae, 'rmse': rmse, 'avg_total': avg_total}


# Main model is now SimulationBasedPredictor (imported from simulation_models)
# Keeping this wrapper for backward compatibility
class NLLPredictionModel(SimulationBasedPredictor):
    """
    Main NLL prediction model using simulation approach
    Wraps SimulationBasedPredictor for backward compatibility
    """

    def __init__(self, config_path=None):
        """Initialize with simulation model"""
        super().__init__(config_path)
        print("NLL Prediction Model initialized with simulation approach")

    def train_and_evaluate(self, features_data, train_seasons=None, test_seasons=None):
        """
        Complete training and evaluation pipeline

        Args:
            features_data: Path to features.json or list of feature dicts
            train_seasons: List of season IDs for training (e.g., [221, 222, 223])
            test_seasons: List of season IDs for testing (e.g., [224])

        Returns:
            Dict with training results and evaluation metrics
        """
        # Load features if path provided
        if isinstance(features_data, (str, Path)):
            with open(features_data, 'r') as f:
                data = json.load(f)
                # Handle nested structure
                if isinstance(data, dict) and 'features' in data:
                    features = data['features']
                else:
                    features = data
        else:
            features = features_data

        print(f"\nLoaded {len(features)} matches with features")

        # Split data by season if seasons provided
        if train_seasons and test_seasons:
            splits = DataSplitter.temporal_split_by_season(
                features,
                train_seasons,
                test_seasons
            )
        else:
            # Use default temporal split
            splits = DataSplitter.temporal_split(features, train_ratio=0.7, val_ratio=0.15)

        train_data = splits['train']
        test_data = splits.get('test', [])

        print(f"Train set: {len(train_data)} matches")
        print(f"Test set: {len(test_data)} matches")

        # Train simulation model (loads team parameters)
        parameters_path = Path(__file__).parent.parent / 'data' / 'team_parameters.json'
        self.train(parameters_path=parameters_path)

        # Evaluate if test data available
        if test_data:
            print("\nEvaluating on test set...")

            # Prepare test data for prediction
            test_inputs = []
            actuals = []

            for match in test_data:
                test_inputs.append({
                    'match_id': match.get('match_id'),
                    'home_team_id': match.get('home_team_id'),
                    'away_team_id': match.get('away_team_id'),
                    'context': {
                        'back_to_back': match.get('home_back_to_back', False) or match.get('away_back_to_back', False),
                        'streak': match.get('home_streak', 0)
                    }
                })

                actuals.append({
                    'winner': 'home' if match.get('home_win', 0) == 1 else 'away',
                    'spread': match.get('spread', 0),
                    'total': match.get('total', 0)
                })

            # Get predictions
            predictions = self.predict(test_inputs)

            # Calculate metrics
            metrics = self.evaluate(test_inputs, actuals)

            # Calculate calibration
            calibration = self.calculate_calibration(predictions, actuals, n_bins=10)

            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"Moneyline Accuracy: {metrics['moneyline_accuracy_pct']:.2f}%")
            print(f"Log Loss: {metrics['log_loss']:.4f}")
            print(f"Brier Score: {metrics['brier_score']:.4f}")
            print(f"Expected Calibration Error: {calibration['expected_calibration_error']:.4f}")
            print(f"Spread MAE: {metrics['spread_mae']:.2f} goals")
            print(f"Total MAE: {metrics['total_mae']:.2f} goals")
            print(f"Test matches: {metrics['n_predictions']}")
            print("=" * 60)

            # Print calibration curve
            print("\nPROBABILITY CALIBRATION ANALYSIS")
            print("=" * 60)
            print(f"{'Predicted Range':<20} {'Actual Win Rate':<20} {'Count':<10} {'Error':<10}")
            print("-" * 60)

            for bin_data in calibration['calibration_curve']:
                pred_range = bin_data['bin_range']
                actual = bin_data['actual_win_rate']
                count = bin_data['count']
                error = bin_data['calibration_error']

                print(f"{pred_range:<20} {actual:<20.3f} {count:<10} {error:<10.3f}")

            print("=" * 60)

            return {
                'model_type': 'simulation',
                'train_size': len(train_data),
                'test_size': len(test_data),
                'metrics': metrics,
                'calibration': calibration,
                'predictions': predictions
            }
        else:
            return {
                'model_type': 'simulation',
                'train_size': len(train_data),
                'test_size': 0,
                'message': 'No test data available for evaluation'
            }


def train_and_save_model(features_path, output_path, train_seasons=None, test_seasons=None):
    """
    Train model and save results

    Args:
        features_path: Path to features.json
        output_path: Path to save model results
        train_seasons: List of season IDs for training
        test_seasons: List of season IDs for testing
    """
    print("=" * 60)
    print("NLL Simulation Model Training")
    print("=" * 60)

    # Initialize model
    config_path = Path(__file__).parent.parent / 'data' / 'simulation_config.json'
    model = NLLPredictionModel(config_path)

    # Train and evaluate
    results = model.train_and_evaluate(
        features_path,
        train_seasons=train_seasons,
        test_seasons=test_seasons
    )

    # Save results
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        # Remove predictions from saved results (too large)
        save_results = {k: v for k, v in results.items() if k != 'predictions'}
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    """Main execution"""
    features_path = Path(__file__).parent.parent / 'data' / 'features.json'
    output_path = Path(__file__).parent.parent / 'data' / 'simulation_results.json'

    # Train on seasons 221-223 (2021-2024), test on season 224 (2024-2025)
    results = train_and_save_model(
        features_path,
        output_path,
        train_seasons=[221, 222, 223],
        test_seasons=[224]
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()

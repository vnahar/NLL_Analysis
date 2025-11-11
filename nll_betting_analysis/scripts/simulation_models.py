"""
NLL Betting Analysis - Simulation Models
Period-based negative binomial simulation for match prediction
"""

import json
import random
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class NegativeBinomialSimulator:
    """
    Simulation-based predictor using period-level negative binomial distribution
    Models lacrosse scoring as overdispersed count process
    """

    def __init__(self, config_path=None):
        """
        Initialize simulator with configuration

        Args:
            config_path: Path to simulation_config.json
        """
        self.team_parameters = {}
        self.league_averages = {}
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        """Load configuration or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                # Flatten nested config structure
                config = {}
                if 'simulation_parameters' in data:
                    config.update(data['simulation_parameters'])
                if 'contextual_adjustments' in data:
                    config.update(data['contextual_adjustments'])
                if 'estimation_parameters' in data:
                    config.update(data['estimation_parameters'])
                # If no nested structure, use as-is
                if not config:
                    config = data
                return config

        # Default configuration
        return {
            'n_simulations': 10000,
            'dispersion_parameter': 2.0,
            'home_advantage_multiplier': 1.07,
            'back_to_back_away_penalty': 0.85,
            'back_to_back_home_bonus': 1.05,
            'streak_effect_per_game': 0.03,
            'max_streak_effect': 0.10,
            'regression_to_mean_weight': 0.3,
            'minimum_games_for_estimate': 5
        }

    def load_team_parameters(self, parameters_path):
        """
        Load team offensive/defensive parameters

        Args:
            parameters_path: Path to team_parameters.json
        """
        with open(parameters_path, 'r') as f:
            data = json.load(f)

        self.team_parameters = data.get('teams', {})
        self.league_averages = data.get('league_averages', {})

        print(f"Loaded parameters for {len(self.team_parameters)} teams")

    def get_team_rates(self, team_id, is_home=True):
        """
        Get offensive/defensive rates for a team

        Args:
            team_id: Team identifier
            is_home: Whether team is playing at home

        Returns:
            Dict with offensive_rate and defensive_rate per quarter
        """
        team_id_str = str(team_id)

        if team_id_str not in self.team_parameters:
            # Use league average for unknown teams
            print(f"Warning: Team {team_id} not found, using league averages")
            return {
                'offensive_rate': self.league_averages.get('avg_offensive_rate', 2.8),
                'defensive_rate': self.league_averages.get('avg_defensive_rate', 2.8),
                'games_analyzed': 0
            }

        params = self.team_parameters[team_id_str]

        # Use home/away specific rates if available
        if is_home and 'home_offensive_rate' in params:
            offensive_rate = params['home_offensive_rate']
        elif not is_home and 'away_offensive_rate' in params:
            offensive_rate = params['away_offensive_rate']
        else:
            offensive_rate = params['offensive_rate_per_quarter']

        # Apply regression to mean for teams with few games
        games = params.get('games_analyzed', 0)
        min_games = self.config['minimum_games_for_estimate']

        if games < min_games:
            league_avg = self.league_averages.get('avg_offensive_rate', 2.8)
            weight = games / min_games
            offensive_rate = weight * offensive_rate + (1 - weight) * league_avg

        return {
            'offensive_rate': offensive_rate,
            'defensive_rate': params.get('defensive_rate_per_quarter', 2.8),
            'games_analyzed': games
        }

    def apply_contextual_adjustments(self, base_rate, context):
        """
        Apply contextual adjustments to base rate

        Args:
            base_rate: Base offensive rate
            context: Dict with is_home, back_to_back, streak

        Returns:
            Adjusted rate
        """
        adjusted_rate = base_rate

        # Home advantage
        if context.get('is_home'):
            adjusted_rate *= self.config['home_advantage_multiplier']

        # Back-to-back effects
        if context.get('back_to_back'):
            if context.get('is_home'):
                adjusted_rate *= self.config['back_to_back_home_bonus']
            else:
                adjusted_rate *= self.config['back_to_back_away_penalty']

        # Streak effects
        streak = context.get('streak', 0)
        if streak != 0:
            streak_effect = min(
                abs(streak) * self.config['streak_effect_per_game'],
                self.config['max_streak_effect']
            )
            if streak > 0:
                adjusted_rate *= (1.0 + streak_effect)
            else:
                adjusted_rate *= (1.0 - streak_effect)

        return adjusted_rate

    @staticmethod
    def gamma_sample(shape, scale):
        """
        Sample from Gamma distribution using Marsaglia and Tsang method

        Args:
            shape: Shape parameter (alpha)
            scale: Scale parameter (beta)

        Returns:
            Sample from Gamma(shape, scale)
        """
        if shape < 1:
            # For shape < 1, use rejection method
            while True:
                u = random.random()
                b = (math.e + shape) / math.e
                p = b * u

                if p <= 1:
                    x = p ** (1.0 / shape)
                    if random.random() <= math.exp(-x):
                        return x * scale
                else:
                    x = -math.log((b - p) / shape)
                    if random.random() <= x ** (shape - 1):
                        return x * scale
        else:
            # Marsaglia and Tsang's method for shape >= 1
            d = shape - 1.0 / 3.0
            c = 1.0 / math.sqrt(9.0 * d)

            while True:
                x = random.gauss(0, 1)
                v = (1.0 + c * x) ** 3

                if v <= 0:
                    continue

                u = random.random()
                if u < 1.0 - 0.0331 * x ** 4:
                    return d * v * scale

                if math.log(u) < 0.5 * x ** 2 + d * (1.0 - v + math.log(v)):
                    return d * v * scale

    @staticmethod
    def poisson_sample(lam):
        """
        Sample from Poisson distribution

        Args:
            lam: Rate parameter (lambda)

        Returns:
            Sample from Poisson(lam)
        """
        if lam < 30:
            # Knuth's method for small lambda
            L = math.exp(-lam)
            k = 0
            p = 1.0

            while p > L:
                k += 1
                p *= random.random()

            return k - 1
        else:
            # Normal approximation for large lambda
            return max(0, int(random.gauss(lam, math.sqrt(lam)) + 0.5))

    def sample_goals(self, offensive_rate, defensive_rate):
        """
        Sample goals using negative binomial distribution

        Models goals as overdispersed count (variance > mean)
        Uses Gamma-Poisson mixture representation

        Args:
            offensive_rate: Team's offensive strength (goals/quarter)
            defensive_rate: Opponent's defensive strength (goals allowed/quarter)

        Returns:
            Integer number of goals scored this period
        """
        # Expected goals = offensive × defensive interaction
        # In lacrosse, better defense (higher defensive_rate means worse defense)
        # so we multiply rates directly
        expected = offensive_rate * (defensive_rate / self.league_averages.get('avg_defensive_rate', 2.8))

        # Negative binomial via Gamma-Poisson mixture
        # NB(mean, r) = Poisson(λ) where λ ~ Gamma(r, mean/r)
        dispersion = self.config['dispersion_parameter']

        # Sample rate from Gamma distribution
        shape = dispersion
        scale = expected / dispersion
        rate = self.gamma_sample(shape, scale)

        # Sample goals from Poisson(rate)
        goals = self.poisson_sample(rate)

        return goals

    def simulate_period(self, home_rate, away_rate):
        """
        Simulate one period (quarter) of play

        Args:
            home_rate: Home team's adjusted offensive rate
            away_rate: Away team's adjusted offensive rate

        Returns:
            Tuple of (home_goals, away_goals)
        """
        # Get defensive rates (inverse of offensive rates for simplicity)
        # Better approach would be to have separate defensive parameters
        home_defensive = self.league_averages.get('avg_defensive_rate', 2.8)
        away_defensive = self.league_averages.get('avg_defensive_rate', 2.8)

        # Sample goals for each team
        home_goals = self.sample_goals(home_rate, away_defensive)
        away_goals = self.sample_goals(away_rate, home_defensive)

        return home_goals, away_goals

    def simulate_match(self, home_team_id, away_team_id, context=None):
        """
        Simulate single match (4 quarters)

        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            context: Dict with contextual factors

        Returns:
            Dict with home_score, away_score, spread, total
        """
        if context is None:
            context = {}

        # Get base rates
        home_params = self.get_team_rates(home_team_id, is_home=True)
        away_params = self.get_team_rates(away_team_id, is_home=False)

        # Apply contextual adjustments
        home_context = {**context, 'is_home': True}
        away_context = {**context, 'is_home': False}

        home_rate = self.apply_contextual_adjustments(
            home_params['offensive_rate'],
            home_context
        )
        away_rate = self.apply_contextual_adjustments(
            away_params['offensive_rate'],
            away_context
        )

        # Simulate 4 quarters
        home_total = 0
        away_total = 0

        for quarter in range(4):
            # Apply Q4 adjustments
            q_home_rate = home_rate
            q_away_rate = away_rate

            if quarter == 3:  # Q4 (0-indexed)
                # Calculate current margin
                margin = home_total - away_total

                # Get Q4 multipliers from config
                q4_trailing_mult = self.config.get('q4_trailing_multiplier', 1.08)
                q4_leading_mult = self.config.get('q4_leading_multiplier', 0.96)
                q4_close_mult = self.config.get('q4_close_multiplier', 1.03)

                # Apply game state adjustments
                if margin < -2:  # Home trailing by 3+
                    q_home_rate *= q4_trailing_mult  # Aggressive
                    q_away_rate *= q4_leading_mult   # Conservative
                elif margin > 2:  # Home leading by 3+
                    q_home_rate *= q4_leading_mult   # Conservative
                    q_away_rate *= q4_trailing_mult  # Aggressive
                else:  # Close game (within 2 goals)
                    q_home_rate *= q4_close_mult
                    q_away_rate *= q4_close_mult

            home_goals, away_goals = self.simulate_period(q_home_rate, q_away_rate)
            home_total += home_goals
            away_total += away_goals

        return {
            'home_score': home_total,
            'away_score': away_total,
            'spread': home_total - away_total,
            'total': home_total + away_total,
            'home_win': 1 if home_total > away_total else 0
        }

    def run_simulations(self, home_team_id, away_team_id, context=None, n_sims=None):
        """
        Run multiple Monte Carlo simulations

        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            context: Contextual factors
            n_sims: Number of simulations (default from config)

        Returns:
            List of simulation results
        """
        if n_sims is None:
            n_sims = self.config['n_simulations']

        results = []
        for _ in range(n_sims):
            result = self.simulate_match(home_team_id, away_team_id, context)
            results.append(result)

        return results

    def get_probabilities(self, simulations):
        """
        Calculate betting probabilities from simulations

        Args:
            simulations: List of simulation results

        Returns:
            Dict with probabilities for moneyline, spread, total
        """
        n = len(simulations)

        # Moneyline probabilities
        home_wins = sum(s['home_win'] for s in simulations)
        p_home_win = home_wins / n
        p_away_win = 1 - p_home_win

        # Spread distribution
        spreads = [s['spread'] for s in simulations]
        spread_mean = sum(spreads) / n
        spread_variance = sum((x - spread_mean) ** 2 for x in spreads) / n
        spread_std = math.sqrt(spread_variance)

        # Total distribution
        totals = [s['total'] for s in simulations]
        total_mean = sum(totals) / n
        total_variance = sum((x - total_mean) ** 2 for x in totals) / n
        total_std = math.sqrt(total_variance)

        # Calculate percentiles
        spreads_sorted = sorted(spreads)
        totals_sorted = sorted(totals)

        def percentile(data, p):
            """Calculate percentile"""
            k = (len(data) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            d0 = data[int(f)] * (c - k)
            d1 = data[int(c)] * (k - f)
            return d0 + d1

        return {
            'moneyline': {
                'home_win_prob': round(p_home_win, 4),
                'away_win_prob': round(p_away_win, 4),
                'home_win_pct': round(p_home_win * 100, 2)
            },
            'spread': {
                'mean': round(spread_mean, 2),
                'std': round(spread_std, 2),
                'p10': round(percentile(spreads_sorted, 0.10), 1),
                'p25': round(percentile(spreads_sorted, 0.25), 1),
                'p50': round(percentile(spreads_sorted, 0.50), 1),
                'p75': round(percentile(spreads_sorted, 0.75), 1),
                'p90': round(percentile(spreads_sorted, 0.90), 1)
            },
            'total': {
                'mean': round(total_mean, 2),
                'std': round(total_std, 2),
                'p10': round(percentile(totals_sorted, 0.10), 1),
                'p25': round(percentile(totals_sorted, 0.25), 1),
                'p50': round(percentile(totals_sorted, 0.50), 1),
                'p75': round(percentile(totals_sorted, 0.75), 1),
                'p90': round(percentile(totals_sorted, 0.90), 1)
            },
            'raw_simulations': simulations
        }

    def predict_match(self, home_team_id, away_team_id, context=None):
        """
        Main prediction interface - runs simulations and returns probabilities

        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            context: Contextual factors (back_to_back, streak, etc.)

        Returns:
            Dict with betting probabilities
        """
        # Run simulations
        simulations = self.run_simulations(home_team_id, away_team_id, context)

        # Calculate probabilities
        probabilities = self.get_probabilities(simulations)

        # Remove raw simulations to save space (optional)
        probabilities.pop('raw_simulations', None)

        return probabilities


class SimulationBasedPredictor:
    """
    Main predictor class that matches API of previous logistic regression model
    """

    def __init__(self, config_path=None):
        """Initialize predictor"""
        self.simulator = NegativeBinomialSimulator(config_path)
        self.is_trained = False

    def train(self, train_data=None, parameters_path=None):
        """
        Train the model (load team parameters)

        Args:
            train_data: Not used for simulation model (parameters pre-computed)
            parameters_path: Path to team_parameters.json
        """
        if parameters_path is None:
            # Default path
            parameters_path = Path(__file__).parent.parent / 'data' / 'team_parameters.json'

        self.simulator.load_team_parameters(parameters_path)
        self.is_trained = True

        print("Simulation model trained successfully")

    def predict(self, test_data, return_probabilities=True):
        """
        Predict outcomes for test matches

        Args:
            test_data: List of dicts with home_team_id, away_team_id, context
            return_probabilities: If True, return full probability distributions

        Returns:
            List of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions = []

        for match in test_data:
            home_id = match.get('home_team_id')
            away_id = match.get('away_team_id')
            context = match.get('context', {})

            # Get prediction
            prob = self.simulator.predict_match(home_id, away_id, context)

            # Add match info
            prediction = {
                'match_id': match.get('match_id'),
                'home_team_id': home_id,
                'away_team_id': away_id,
                **prob
            }

            predictions.append(prediction)

        return predictions

    def evaluate(self, test_data, actuals):
        """
        Evaluate model performance

        Args:
            test_data: List of test matches
            actuals: List of actual outcomes

        Returns:
            Dict with evaluation metrics
        """
        predictions = self.predict(test_data)

        # Moneyline accuracy
        correct = 0
        log_loss_sum = 0
        brier_score_sum = 0

        for pred, actual in zip(predictions, actuals):
            predicted_winner = 'home' if pred['moneyline']['home_win_prob'] > 0.5 else 'away'
            actual_winner = actual.get('winner')

            if predicted_winner == actual_winner:
                correct += 1

            # Log Loss calculation
            p_home = pred['moneyline']['home_win_prob']
            y_home = 1 if actual_winner == 'home' else 0

            # Clip probabilities to avoid log(0)
            p_home = max(0.001, min(0.999, p_home))

            # Log loss formula: -[y*log(p) + (1-y)*log(1-p)]
            log_loss_value = -(y_home * math.log(p_home) + (1 - y_home) * math.log(1 - p_home))
            log_loss_sum += log_loss_value

            # Brier Score: mean squared error of probabilities
            brier_score = (p_home - y_home) ** 2
            brier_score_sum += brier_score

        accuracy = correct / len(predictions) if predictions else 0
        log_loss = log_loss_sum / len(predictions) if predictions else 0
        brier_score = brier_score_sum / len(predictions) if predictions else 0

        # Spread MAE
        spread_errors = []
        for pred, actual in zip(predictions, actuals):
            predicted_spread = pred['spread']['mean']
            actual_spread = actual.get('spread', 0)
            spread_errors.append(abs(predicted_spread - actual_spread))

        spread_mae = sum(spread_errors) / len(spread_errors) if spread_errors else 0

        # Total MAE
        total_errors = []
        for pred, actual in zip(predictions, actuals):
            predicted_total = pred['total']['mean']
            actual_total = actual.get('total', 0)
            total_errors.append(abs(predicted_total - actual_total))

        total_mae = sum(total_errors) / len(total_errors) if total_errors else 0

        return {
            'moneyline_accuracy': round(accuracy, 4),
            'moneyline_accuracy_pct': round(accuracy * 100, 2),
            'log_loss': round(log_loss, 4),
            'brier_score': round(brier_score, 4),
            'spread_mae': round(spread_mae, 2),
            'total_mae': round(total_mae, 2),
            'n_predictions': len(predictions)
        }

    def calculate_calibration(self, predictions, actuals, n_bins=10):
        """
        Analyze probability calibration

        Args:
            predictions: List of prediction dicts
            actuals: List of actual outcome dicts
            n_bins: Number of bins for calibration curve

        Returns:
            Dict with calibration curve data
        """
        # Group predictions into probability bins
        bins = [[] for _ in range(n_bins)]

        for pred, actual in zip(predictions, actuals):
            p_home = pred['moneyline']['home_win_prob']
            y_home = 1 if actual.get('winner') == 'home' else 0

            # Determine bin (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
            bin_idx = min(int(p_home * n_bins), n_bins - 1)
            bins[bin_idx].append({
                'predicted': p_home,
                'actual': y_home
            })

        # Calculate actual win rate per bin
        calibration_curve = []
        total_samples = sum(len(bin_data) for bin_data in bins if bin_data)
        ece_sum = 0

        for i, bin_data in enumerate(bins):
            if not bin_data:
                continue

            avg_predicted = sum(d['predicted'] for d in bin_data) / len(bin_data)
            avg_actual = sum(d['actual'] for d in bin_data) / len(bin_data)
            error = abs(avg_predicted - avg_actual)

            # Contribution to Expected Calibration Error
            ece_sum += (len(bin_data) / total_samples) * error

            calibration_curve.append({
                'bin': i,
                'bin_range': f'{i/n_bins:.1f}-{(i+1)/n_bins:.1f}',
                'avg_predicted_prob': round(avg_predicted, 3),
                'actual_win_rate': round(avg_actual, 3),
                'count': len(bin_data),
                'calibration_error': round(error, 3)
            })

        return {
            'calibration_curve': calibration_curve,
            'expected_calibration_error': round(ece_sum, 4),
            'n_bins': n_bins
        }

    @staticmethod
    def convert_probability_to_odds(probability):
        """
        Convert probability to American moneyline odds

        Args:
            probability: Win probability (0-1)

        Returns:
            American odds (e.g., -150, +200)
        """
        if probability >= 0.5:
            # Favorite (negative odds)
            odds = -100 * probability / (1 - probability)
        else:
            # Underdog (positive odds)
            odds = 100 * (1 - probability) / probability

        return round(odds)


def main():
    """Main execution for testing"""
    print("=" * 60)
    print("NLL Simulation Model Test")
    print("=" * 60)

    # Initialize predictor
    predictor = SimulationBasedPredictor()

    # Train (load parameters)
    parameters_path = Path(__file__).parent.parent / 'data' / 'team_parameters.json'
    predictor.train(parameters_path=parameters_path)

    # Test prediction
    test_match = {
        'match_id': 'test_001',
        'home_team_id': 542,  # San Diego
        'away_team_id': 543,  # Vancouver
        'context': {
            'back_to_back': False,
            'streak': 2  # Home team on 2-game win streak
        }
    }

    print("\nTest Prediction:")
    print(f"  Home: Team {test_match['home_team_id']}")
    print(f"  Away: Team {test_match['away_team_id']}")
    print(f"  Context: {test_match['context']}")

    prediction = predictor.predict([test_match])[0]

    print("\nResults:")
    print(f"  Moneyline: Home {prediction['moneyline']['home_win_pct']:.1f}% | Away {prediction['moneyline']['away_win_prob']*100:.1f}%")
    print(f"  Spread: {prediction['spread']['mean']:.1f} ± {prediction['spread']['std']:.1f}")
    print(f"  Total: {prediction['total']['mean']:.1f} ± {prediction['total']['std']:.1f}")

    print("\n" + "=" * 60)
    print("Simulation model test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

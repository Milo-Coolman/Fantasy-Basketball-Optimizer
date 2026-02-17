#!/usr/bin/env python3
"""
Train Fantasy Basketball ML Models.

This script loads collected training data and trains ML models for
predicting player statistics:
- Counting stats: PTS, TRB, AST, STL, BLK, TOV, 3P
- Shooting percentages: FG%, FT%

Usage:
    python train_models.py
    python train_models.py --tune  # With hyperparameter tuning (slower)
    python train_models.py --algo random_forest

Reference: PRD Section 6 - Machine Learning Models
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path BEFORE other imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler

# Import TrainedModel and ModelMetrics from ml_model for proper serialization
# This ensures pickled models can be loaded by ml_model.py
from backend.projections.ml_model import TrainedModel, ModelMetrics


# =============================================================================
# Setup Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Directory paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(MODULE_DIR, 'trained_models')
TRAINING_DATA_DIR = os.path.join(MODULE_DIR, 'training_data')
TRAINING_DATA_FILE = os.path.join(TRAINING_DATA_DIR, 'nba_training_data.csv')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Stats to model
COUNTING_STATS = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', '3p']
SHOOTING_STATS = ['fg_pct', 'ft_pct']

# Features for counting stats
COUNTING_FEATURES = [
    # Demographics
    'age',
    # Playing time
    'mp', 'g', 'gs',
    # Usage and efficiency
    'usg_pct', 'ts_pct', 'per',
    # Shooting profile
    'efg_pct', '3p_pct', 'ft_pct',
    # Advanced metrics
    'bpm', 'vorp', 'ws',
    # Per-36 stats
    'pts_per36', 'trb_per36', 'ast_per36', 'stl_per36', 'blk_per36',
    # Previous season stats
    'pts_prev_season', 'trb_prev_season', 'ast_prev_season',
    'mp_prev_season', 'g_prev_season',
    # Position encoding
    'is_guard', 'is_forward', 'is_center',
    # Additional features
    '3par', 'ftr', 'orb_pct', 'drb_pct', 'ast_pct', 'stl_pct', 'blk_pct',
]

# Features for shooting stats
SHOOTING_FEATURES = [
    'age', 'mp', 'g', 'fga', 'fta',
    '3p_rate', '3par', 'ftr',
    'is_guard', 'is_forward', 'is_center',
    'ts_pct', 'efg_pct',
    'per', 'usg_pct',
]

# Train/test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42


# =============================================================================
# Helper Functions
# =============================================================================

def print_banner(text: str, char: str = '=', width: int = 70) -> None:
    """Print a formatted banner."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_section(text: str, char: str = '-', width: int = 50) -> None:
    """Print a section header."""
    print()
    print(f"{char * 3} {text} {char * 3}")


def format_time(seconds: float) -> str:
    """Format seconds into a readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_training_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess training data."""
    print_section("Loading Training Data")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Training data not found: {filepath}\n"
            "Run collect_training_data.py first to generate training data."
        )

    print(f"  Loading: {filepath}")
    df = pd.read_csv(filepath)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    # Add engineered features
    df = engineer_features(df)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataset."""
    print_section("Engineering Features")
    df = df.copy()

    added_features = []

    # Games started percentage
    if 'gs' in df.columns and 'g' in df.columns:
        df['gs_pct'] = (df['gs'] / df['g'].clip(lower=1)).clip(0, 1)
        added_features.append('gs_pct')

    # 3-point rate (proportion of shots that are 3s)
    if '3pa' in df.columns and 'fga' in df.columns:
        df['3p_rate'] = (df['3pa'] / df['fga'].clip(lower=1)).clip(0, 1)
        added_features.append('3p_rate')

    # Free throw rate
    if 'fta' in df.columns and 'fga' in df.columns:
        df['ft_rate'] = (df['fta'] / df['fga'].clip(lower=1)).clip(0, 3)
        added_features.append('ft_rate')

    # Position encoding
    if 'pos' in df.columns:
        df['is_guard'] = df['pos'].str.contains('G', na=False).astype(int)
        df['is_forward'] = df['pos'].str.contains('F', na=False).astype(int)
        df['is_center'] = df['pos'].str.contains('C', na=False).astype(int)
        added_features.extend(['is_guard', 'is_forward', 'is_center'])

    # Experience (years since age 19)
    if 'age' in df.columns:
        df['experience'] = (df['age'] - 19).clip(lower=0)
        df['is_prime'] = ((df['age'] >= 24) & (df['age'] <= 30)).astype(int)
        df['age_squared'] = df['age'] ** 2
        added_features.extend(['experience', 'is_prime', 'age_squared'])

    # Per-36 minute stats (if not already present)
    counting_stats = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', '3p']
    for stat in counting_stats:
        per36_col = f'{stat}_per36'
        if per36_col not in df.columns and stat in df.columns and 'mp' in df.columns:
            df[per36_col] = np.where(
                df['mp'] > 0,
                (df[stat] / df['mp']) * 36,
                0
            )
            added_features.append(per36_col)

    # Previous season stats (grouped by player)
    if 'player_normalized' in df.columns and 'season' in df.columns:
        df = df.sort_values(['player_normalized', 'season'])
        for stat in ['pts', 'trb', 'ast', 'mp', 'g', 'fg_pct', 'ft_pct']:
            if stat in df.columns:
                prev_col = f'{stat}_prev_season'
                df[prev_col] = df.groupby('player_normalized')[stat].shift(1)
                added_features.append(prev_col)

    print(f"  Added {len(added_features)} engineered features:")
    for feat in added_features[:10]:
        print(f"    - {feat}")
    if len(added_features) > 10:
        print(f"    ... and {len(added_features) - 10} more")

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_list: List[str],
    target: str
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare features and target for model training."""
    # Filter to available features
    available_features = [f for f in feature_list if f in df.columns]

    # Remove target from features if present
    available_features = [f for f in available_features if f != target]

    # Get feature matrix and target
    X = df[available_features].copy()
    y = df[target].copy()

    # Remove rows where target is missing
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]

    # Fill missing feature values with median
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)

    return X, y, available_features


# =============================================================================
# Model Training
# =============================================================================

def train_counting_stat_model(
    df: pd.DataFrame,
    stat: str,
    algorithm: str = 'gradient_boosting',
    tune_hyperparams: bool = False
) -> Dict[str, Any]:
    """Train a model for a counting statistic."""
    print_section(f"Training {stat.upper()} Model")
    start_time = time.time()

    print(f"  Algorithm: {algorithm}")
    print(f"  Tuning: {'Yes' if tune_hyperparams else 'No'}")

    # Prepare data
    X, y, features = prepare_features_and_target(df, COUNTING_FEATURES, stat)
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(features)}")

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_STATE
    )

    print(f"  Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Initialize model
    if algorithm == 'gradient_boosting':
        base_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        )
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
    else:  # random_forest
        base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
        }

    # Train model
    if tune_hyperparams:
        print("  Performing hyperparameter search...")
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"  Best params: {best_params}")
    else:
        model = base_model
        best_params = base_model.get_params()
        print("  Training model...")
        model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n  Validation Results:")
    print(f"    RMSE: {val_rmse:.4f}")
    print(f"    MAE:  {val_mae:.4f}")
    print(f"    R²:   {val_r2:.4f}")

    # Evaluate on test set
    y_test_pred = model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_ev = explained_variance_score(y_test, y_test_pred)

    print(f"\n  Test Results:")
    print(f"    RMSE: {test_rmse:.4f}")
    print(f"    MAE:  {test_mae:.4f}")
    print(f"    R²:   {test_r2:.4f}")
    print(f"    Explained Variance: {test_ev:.4f}")

    # Cross-validation on full training set
    print("\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, scaler.transform(X_train_val), y_train_val,
        cv=5, scoring='neg_root_mean_squared_error'
    )
    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"    CV RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n  Top 5 Features:")
        for i in range(min(5, len(features))):
            idx = indices[i]
            print(f"    {i+1}. {features[idx]}: {importances[idx]:.4f}")

    elapsed = time.time() - start_time
    print(f"\n  Time: {format_time(elapsed)}")

    return {
        'stat': stat,
        'model': model,
        'scaler': scaler,
        'features': features,
        'algorithm': algorithm,
        'hyperparameters': best_params,
        'metrics': {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'explained_variance': test_ev,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std,
        },
        'training_time': elapsed,
    }


def train_shooting_pct_model(
    df: pd.DataFrame,
    stat: str,
    algorithm: str = 'ridge'
) -> Dict[str, Any]:
    """Train a model for a shooting percentage statistic."""
    print_section(f"Training {stat.upper()} Model")
    start_time = time.time()

    print(f"  Algorithm: {algorithm}")

    # Filter data BEFORE preparing features to ensure we have enough volume
    # Note: fga/fta are per-game averages, not totals
    df_filtered = df.copy()

    if stat == 'fg_pct' and 'fga' in df.columns:
        min_fga_per_game = 5.0  # At least 5 field goal attempts per game
        df_filtered = df_filtered[df_filtered['fga'] >= min_fga_per_game]
        print(f"  Filtered to players with {min_fga_per_game}+ FGA/game: {len(df_filtered)} samples")
    elif stat == 'ft_pct' and 'fta' in df.columns:
        min_fta_per_game = 1.0  # At least 1 free throw attempt per game
        df_filtered = df_filtered[df_filtered['fta'] >= min_fta_per_game]
        print(f"  Filtered to players with {min_fta_per_game}+ FTA/game: {len(df_filtered)} samples")

    # Prepare data
    X, y, features = prepare_features_and_target(df_filtered, SHOOTING_FEATURES, stat)

    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(features)}")

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_STATE
    )

    print(f"  Train/Val/Test: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Initialize model
    if algorithm == 'ridge':
        model = Ridge(alpha=1.0)
    elif algorithm == 'lasso':
        model = Lasso(alpha=0.01, max_iter=10000)
    else:  # elastic_net
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)

    # Train
    print("  Training model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_val_pred = np.clip(model.predict(X_val_scaled), 0, 1)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n  Validation Results:")
    print(f"    RMSE: {val_rmse:.4f}")
    print(f"    MAE:  {val_mae:.4f}")
    print(f"    R²:   {val_r2:.4f}")

    # Evaluate on test set
    y_test_pred = np.clip(model.predict(X_test_scaled), 0, 1)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_ev = explained_variance_score(y_test, y_test_pred)

    print(f"\n  Test Results:")
    print(f"    RMSE: {test_rmse:.4f}")
    print(f"    MAE:  {test_mae:.4f}")
    print(f"    R²:   {test_r2:.4f}")
    print(f"    Explained Variance: {test_ev:.4f}")

    # Cross-validation
    print("\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, scaler.transform(X_train_val), y_train_val,
        cv=5, scoring='neg_root_mean_squared_error'
    )
    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"    CV RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})")

    # Feature coefficients
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        print(f"\n  Top 5 Coefficients:")
        for i in range(min(5, len(features))):
            idx = indices[i]
            print(f"    {i+1}. {features[idx]}: {coefs[idx]:.4f}")

    elapsed = time.time() - start_time
    print(f"\n  Time: {format_time(elapsed)}")

    return {
        'stat': stat,
        'model': model,
        'scaler': scaler,
        'features': features,
        'algorithm': algorithm,
        'hyperparameters': {'alpha': getattr(model, 'alpha', None)},
        'metrics': {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'explained_variance': test_ev,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std,
        },
        'training_time': elapsed,
    }


# =============================================================================
# Model Saving
# =============================================================================

def save_model(model_result: Dict[str, Any], output_dir: str) -> str:
    """Save a trained model to disk as a proper TrainedModel object."""
    stat = model_result['stat']
    filepath = os.path.join(output_dir, f"{stat}_model.pkl")

    # Create ModelMetrics object
    metrics_dict = model_result['metrics']
    metrics = ModelMetrics(
        rmse=metrics_dict['rmse'],
        mae=metrics_dict['mae'],
        r2=metrics_dict['r2'],
        explained_variance=metrics_dict['explained_variance'],
        cv_rmse_mean=metrics_dict.get('cv_rmse', 0.0),
        cv_rmse_std=metrics_dict.get('cv_std', 0.0),
    )

    # Create TrainedModel object (this has the predict() method)
    trained_model = TrainedModel(
        name=f"{stat}_model",
        target_stat=stat,
        model=model_result['model'],
        scaler=model_result['scaler'],
        features=model_result['features'],
        metrics=metrics,
        trained_at=datetime.now().isoformat(),
        model_type=model_result['algorithm'],
        hyperparameters=model_result['hyperparameters'],
    )

    # Save the TrainedModel object directly
    trained_model.save(filepath)

    return filepath


def generate_report(results: List[Dict[str, Any]], output_dir: str) -> str:
    """Generate a summary report of model training."""
    lines = [
        "=" * 70,
        "FANTASY BASKETBALL ML MODEL TRAINING REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Models Directory: {output_dir}",
        "",
    ]

    # Summary table
    lines.append("MODEL PERFORMANCE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"{'Stat':<10} {'Algorithm':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Time':<10}")
    lines.append("-" * 70)

    total_time = 0
    for r in results:
        stat = r['stat'].upper()
        algo = r['algorithm']
        rmse = f"{r['metrics']['rmse']:.4f}"
        mae = f"{r['metrics']['mae']:.4f}"
        r2 = f"{r['metrics']['r2']:.4f}"
        time_str = format_time(r['training_time'])
        total_time += r['training_time']

        lines.append(f"{stat:<10} {algo:<20} {rmse:<10} {mae:<10} {r2:<10} {time_str:<10}")

    lines.append("-" * 70)
    lines.append(f"Total Training Time: {format_time(total_time)}")
    lines.append("")

    # Detailed metrics for each model
    lines.append("\nDETAILED METRICS BY MODEL")
    lines.append("=" * 70)

    for r in results:
        stat = r['stat'].upper()
        lines.append(f"\n{stat}")
        lines.append("-" * 40)
        lines.append(f"  Algorithm:          {r['algorithm']}")
        lines.append(f"  Features:           {len(r['features'])}")
        lines.append(f"  Test RMSE:          {r['metrics']['rmse']:.4f}")
        lines.append(f"  Test MAE:           {r['metrics']['mae']:.4f}")
        lines.append(f"  Test R²:            {r['metrics']['r2']:.4f}")
        lines.append(f"  Explained Variance: {r['metrics']['explained_variance']:.4f}")
        lines.append(f"  CV RMSE:            {r['metrics']['cv_rmse']:.4f} (+/- {r['metrics']['cv_std']:.4f})")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    return report_text


def save_metadata(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Save training metadata as JSON."""
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'models': {},
        'counting_stats': COUNTING_STATS,
        'shooting_stats': SHOOTING_STATS,
    }

    for r in results:
        metadata['models'][r['stat']] = {
            'path': f"{r['stat']}_model.pkl",
            'algorithm': r['algorithm'],
            'features': r['features'],
            'metrics': r['metrics'],
            'hyperparameters': r['hyperparameters'],
            'training_time_seconds': r['training_time'],
        }

    metadata_path = os.path.join(output_dir, 'models_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {metadata_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(
        description='Train Fantasy Basketball ML Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_models.py                          # Train with defaults
    python train_models.py --tune                   # With hyperparameter tuning
    python train_models.py --algo random_forest     # Use Random Forest
    python train_models.py --shooting-algo lasso    # Use Lasso for shooting %
        """
    )
    parser.add_argument(
        '--data',
        type=str,
        default=TRAINING_DATA_FILE,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--algo', '--counting-algo',
        type=str,
        choices=['gradient_boosting', 'random_forest'],
        default='gradient_boosting',
        dest='counting_algo',
        help='Algorithm for counting stats (default: gradient_boosting)'
    )
    parser.add_argument(
        '--shooting-algo',
        type=str,
        choices=['ridge', 'lasso', 'elastic_net'],
        default='ridge',
        help='Algorithm for shooting percentages (default: ridge)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (slower but potentially better)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=MODELS_DIR,
        help=f'Directory to save trained models (default: {MODELS_DIR})'
    )
    parser.add_argument(
        '--stats',
        type=str,
        nargs='+',
        choices=COUNTING_STATS + SHOOTING_STATS,
        help='Specific stats to train (default: all)'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Print header
    print_banner("FANTASY BASKETBALL ML MODEL TRAINING")
    print(f"  Training Data:     {args.data}")
    print(f"  Counting Algorithm: {args.counting_algo}")
    print(f"  Shooting Algorithm: {args.shooting_algo}")
    print(f"  Hyperparameter Tuning: {'Yes' if args.tune else 'No'}")
    print(f"  Output Directory:  {args.output_dir}")

    overall_start = time.time()

    try:
        # Load training data
        df = load_training_data(args.data)

        # Determine which stats to train
        counting_to_train = [s for s in COUNTING_STATS if s in df.columns]
        shooting_to_train = [s for s in SHOOTING_STATS if s in df.columns]

        if args.stats:
            counting_to_train = [s for s in counting_to_train if s in args.stats]
            shooting_to_train = [s for s in shooting_to_train if s in args.stats]

        print(f"\n  Stats to train:")
        print(f"    Counting: {counting_to_train}")
        print(f"    Shooting: {shooting_to_train}")

        # Train models
        all_results = []

        # Train counting stat models
        if counting_to_train:
            print_banner("TRAINING COUNTING STAT MODELS", char='=')
            for stat in counting_to_train:
                try:
                    result = train_counting_stat_model(
                        df, stat, args.counting_algo, args.tune
                    )
                    all_results.append(result)

                    # Save model
                    filepath = save_model(result, args.output_dir)
                    print(f"  Saved: {filepath}")

                except Exception as e:
                    logger.error(f"Failed to train {stat} model: {e}")
                    import traceback
                    traceback.print_exc()

        # Train shooting percentage models
        if shooting_to_train:
            print_banner("TRAINING SHOOTING PERCENTAGE MODELS", char='=')
            for stat in shooting_to_train:
                try:
                    result = train_shooting_pct_model(
                        df, stat, args.shooting_algo
                    )
                    all_results.append(result)

                    # Save model
                    filepath = save_model(result, args.output_dir)
                    print(f"  Saved: {filepath}")

                except Exception as e:
                    logger.error(f"Failed to train {stat} model: {e}")
                    import traceback
                    traceback.print_exc()

        # Generate and print report
        print_banner("TRAINING COMPLETE")

        if all_results:
            # Save metadata
            print_section("Saving Metadata")
            save_metadata(all_results, args.output_dir)

            # Generate report
            print_section("Generating Report")
            report = generate_report(all_results, args.output_dir)
            print(f"  Saved: {os.path.join(args.output_dir, 'training_report.txt')}")

            # Print summary
            print("\n" + report)

        overall_elapsed = time.time() - overall_start
        print(f"\nTotal Time: {format_time(overall_elapsed)}")
        print(f"Models saved to: {args.output_dir}")

        # List saved files
        print("\nSaved Files:")
        for f in sorted(os.listdir(args.output_dir)):
            filepath = os.path.join(args.output_dir, f)
            size = os.path.getsize(filepath)
            print(f"  {f}: {size:,} bytes")

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

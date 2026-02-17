#!/usr/bin/env python3
"""
Machine Learning Models for Fantasy Basketball Projections.

This module implements ML models for predicting player statistics
including counting stats (points, rebounds, assists, etc.) and
shooting percentages (FG%, FT%).

Models:
- Gradient Boosting / Random Forest for counting stats
- Ridge Regression for shooting percentages

Reference: PRD Section 6 - Machine Learning Models
"""

import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
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

# Default training data file
DEFAULT_TRAINING_DATA = os.path.join(TRAINING_DATA_DIR, 'nba_training_data.csv')

# Counting stats to model (separate model for each)
COUNTING_STATS = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', '3p']

# Shooting percentage stats to model
SHOOTING_STATS = ['fg_pct', 'ft_pct']

# All target stats
ALL_STATS = COUNTING_STATS + SHOOTING_STATS

# Features for counting stats models
COUNTING_FEATURES = [
    # Demographics
    'age',
    # Playing time
    'mp', 'g', 'gs_pct',
    # Usage and efficiency
    'usg_pct', 'ts_pct', 'per',
    # Shooting profile
    '3p_rate', 'ft_rate', 'efg_pct',
    # Position encoding
    'is_guard', 'is_forward', 'is_center',
    # Advanced metrics
    'bpm', 'vorp', 'ws',
    # Per-36 stats (normalize for minutes)
    'pts_per36', 'trb_per36', 'ast_per36', 'stl_per36', 'blk_per36',
    # Previous season (if available)
    'pts_prev_season', 'trb_prev_season', 'ast_prev_season',
    'mp_prev_season', 'g_prev_season',
]

# Features for shooting percentage models
SHOOTING_FEATURES = [
    # Demographics
    'age',
    # Volume
    'fga', 'fta', 'mp', 'g',
    # Shot profile
    '3p_rate', 'ft_rate',
    # Position
    'is_guard', 'is_forward', 'is_center',
    # Previous performance
    'fg_pct_prev_season' if 'fg_pct_prev_season' in [] else 'fg_pct',
    'ft_pct_prev_season' if 'ft_pct_prev_season' in [] else 'ft_pct',
    # Efficiency
    'ts_pct', 'efg_pct',
]

# Train/validation/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random state for reproducibility
RANDOM_STATE = 42


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained model."""
    rmse: float
    mae: float
    r2: float
    explained_variance: float
    cv_rmse_mean: float = 0.0
    cv_rmse_std: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'rmse': round(self.rmse, 4),
            'mae': round(self.mae, 4),
            'r2': round(self.r2, 4),
            'explained_variance': round(self.explained_variance, 4),
            'cv_rmse_mean': round(self.cv_rmse_mean, 4),
            'cv_rmse_std': round(self.cv_rmse_std, 4),
        }


@dataclass
class TrainedModel:
    """Container for a trained model with metadata."""
    name: str
    target_stat: str
    model: Any
    scaler: Optional[StandardScaler]
    features: List[str]
    metrics: ModelMetrics
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model_type: str = "gradient_boosting"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        # Create a copy with all required features, filling missing ones with 0
        X_ordered = pd.DataFrame(index=X.index)

        for feature in self.features:
            if feature in X.columns:
                X_ordered[feature] = X[feature]
            else:
                X_ordered[feature] = 0  # Default missing features to 0

        # Fill any NaN values
        X_ordered = X_ordered.fillna(0)

        # Scale if scaler exists
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_ordered)
        else:
            X_scaled = X_ordered.values

        return self.model.predict(X_scaled)

    def save(self, filepath: str) -> None:
        """Save the model to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TrainedModel':
        """Load a model from a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# Feature Engineering
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    feature_list: List[str],
    target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.

    Args:
        df: Training DataFrame
        feature_list: List of feature column names
        target: Target column name

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Filter to available features
    available_features = [f for f in feature_list if f in df.columns]

    if len(available_features) < len(feature_list) * 0.5:
        logger.warning(
            f"Only {len(available_features)}/{len(feature_list)} features available"
        )

    # Get feature matrix
    X = df[available_features].copy()

    # Get target
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame")
    y = df[target].copy()

    # Remove rows where target is missing
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]

    # Fill missing feature values
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype in [np.float64, np.int64]:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)

    logger.info(f"Prepared {len(X)} samples with {len(available_features)} features for '{target}'")

    return X, y


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional engineered features to the dataset.

    Args:
        df: Training DataFrame

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Experience proxy (seasons in league)
    if 'age' in df.columns and 'season' in df.columns:
        # Assume players enter at age 19-20
        df['experience'] = df['age'] - 19
        df['experience'] = df['experience'].clip(0, 20)

    # Minutes opportunity (games started * minutes per game)
    if 'gs' in df.columns and 'mp' in df.columns:
        df['minutes_opportunity'] = df['gs'] * df['mp']

    # Scoring dominance (points as % of team average ~110)
    if 'pts' in df.columns:
        df['scoring_share'] = df['pts'] / 110  # Approximate team avg

    # Versatility score (combined stocks + assists)
    if all(col in df.columns for col in ['ast', 'stl', 'blk']):
        df['versatility'] = df['ast'] + df['stl'] * 2 + df['blk'] * 2

    # Shot selection efficiency
    if '3p_pct' in df.columns and 'fg_pct' in df.columns:
        df['shot_selection'] = df['efg_pct'] if 'efg_pct' in df.columns else df['fg_pct']

    # Age curve features (peak at 27)
    if 'age' in df.columns:
        df['age_from_peak'] = abs(df['age'] - 27)
        df['is_prime'] = ((df['age'] >= 24) & (df['age'] <= 30)).astype(int)

    return df


# =============================================================================
# Model Training
# =============================================================================

class FantasyBasketballMLModel:
    """
    Machine Learning model manager for fantasy basketball projections.

    Handles training, evaluation, and persistence of multiple stat models.
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the ML model manager.

        Args:
            models_dir: Directory for saving trained models
        """
        self.models_dir = models_dir or MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

        self.counting_models: Dict[str, TrainedModel] = {}
        self.shooting_models: Dict[str, TrainedModel] = {}
        self._training_history: List[Dict] = []

    def load_training_data(self, filepath: str = DEFAULT_TRAINING_DATA) -> pd.DataFrame:
        """
        Load and preprocess training data.

        Args:
            filepath: Path to training data CSV

        Returns:
            Preprocessed DataFrame
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Training data not found: {filepath}\n"
                "Run collect_training_data.py first to generate training data."
            )

        logger.info(f"Loading training data from: {filepath}")
        df = pd.read_csv(filepath)

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Add engineered features
        df = add_engineered_features(df)

        return df

    def train_counting_stat_model(
        self,
        df: pd.DataFrame,
        stat: str,
        algorithm: str = 'gradient_boosting',
        tune_hyperparams: bool = False
    ) -> TrainedModel:
        """
        Train a model for a counting stat.

        Args:
            df: Training DataFrame
            stat: Target statistic (e.g., 'pts', 'trb', 'ast')
            algorithm: 'gradient_boosting' or 'random_forest'
            tune_hyperparams: Whether to perform grid search

        Returns:
            Trained model object
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for: {stat.upper()}")
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"{'='*60}")

        # Prepare features
        X, y = prepare_features(df, COUNTING_FEATURES, stat)
        available_features = X.columns.tolist()

        # Split data: train/val/test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            random_state=RANDOM_STATE
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

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
                'min_samples_leaf': [3, 5, 10],
            }

        # Hyperparameter tuning
        if tune_hyperparams:
            logger.info("Performing hyperparameter search...")
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='neg_root_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
        else:
            model = base_model
            best_params = model.get_params()
            model.fit(X_train_scaled, y_train)

        # Validation predictions
        y_val_pred = model.predict(X_val_scaled)

        # Evaluate on validation set
        val_metrics = self._calculate_metrics(y_val, y_val_pred, model, X_train_scaled, y_train)

        logger.info(f"\nValidation Metrics for {stat.upper()}:")
        logger.info(f"  RMSE: {val_metrics.rmse:.4f}")
        logger.info(f"  MAE:  {val_metrics.mae:.4f}")
        logger.info(f"  R²:   {val_metrics.r2:.4f}")
        logger.info(f"  CV RMSE: {val_metrics.cv_rmse_mean:.4f} (+/- {val_metrics.cv_rmse_std:.4f})")

        # Final evaluation on test set
        y_test_pred = model.predict(X_test_scaled)
        test_metrics = self._calculate_metrics(y_test, y_test_pred, model, X_train_scaled, y_train)

        logger.info(f"\nTest Metrics for {stat.upper()}:")
        logger.info(f"  RMSE: {test_metrics.rmse:.4f}")
        logger.info(f"  MAE:  {test_metrics.mae:.4f}")
        logger.info(f"  R²:   {test_metrics.r2:.4f}")

        # Create trained model object
        trained_model = TrainedModel(
            name=f"{stat}_model",
            target_stat=stat,
            model=model,
            scaler=scaler,
            features=available_features,
            metrics=test_metrics,
            model_type=algorithm,
            hyperparameters=best_params
        )

        # Store model
        self.counting_models[stat] = trained_model

        # Log feature importance
        self._log_feature_importance(model, available_features, stat, algorithm)

        return trained_model

    def train_shooting_pct_model(
        self,
        df: pd.DataFrame,
        stat: str,
        algorithm: str = 'ridge'
    ) -> TrainedModel:
        """
        Train a model for a shooting percentage stat.

        Uses Ridge/Lasso regression for more stable predictions on percentages.

        Args:
            df: Training DataFrame
            stat: Target statistic ('fg_pct' or 'ft_pct')
            algorithm: 'ridge', 'lasso', or 'elastic_net'

        Returns:
            Trained model object
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training shooting model for: {stat.upper()}")
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"{'='*60}")

        # Use shooting-specific features
        features = [f for f in SHOOTING_FEATURES if f in df.columns and f != stat]

        # Prepare features
        X, y = prepare_features(df, features, stat)
        available_features = X.columns.tolist()

        # Filter out samples with 0 attempts (no valid shooting percentage)
        if stat == 'fg_pct':
            valid_idx = df.loc[X.index, 'fga'] > 50 if 'fga' in df.columns else pd.Series(True, index=X.index)
        else:  # ft_pct
            valid_idx = df.loc[X.index, 'fta'] > 20 if 'fta' in df.columns else pd.Series(True, index=X.index)

        X = X[valid_idx]
        y = y[valid_idx]

        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            random_state=RANDOM_STATE
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Initialize model
        if algorithm == 'ridge':
            model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        elif algorithm == 'lasso':
            model = Lasso(alpha=0.01, random_state=RANDOM_STATE)
        else:  # elastic_net
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE)

        # Train
        model.fit(X_train_scaled, y_train)

        # Validation
        y_val_pred = model.predict(X_val_scaled)
        # Clip predictions to valid percentage range
        y_val_pred = np.clip(y_val_pred, 0, 1)

        val_metrics = self._calculate_metrics(y_val, y_val_pred, model, X_train_scaled, y_train)

        logger.info(f"\nValidation Metrics for {stat.upper()}:")
        logger.info(f"  RMSE: {val_metrics.rmse:.4f}")
        logger.info(f"  MAE:  {val_metrics.mae:.4f}")
        logger.info(f"  R²:   {val_metrics.r2:.4f}")

        # Test evaluation
        y_test_pred = model.predict(X_test_scaled)
        y_test_pred = np.clip(y_test_pred, 0, 1)
        test_metrics = self._calculate_metrics(y_test, y_test_pred, model, X_train_scaled, y_train)

        logger.info(f"\nTest Metrics for {stat.upper()}:")
        logger.info(f"  RMSE: {test_metrics.rmse:.4f}")
        logger.info(f"  MAE:  {test_metrics.mae:.4f}")
        logger.info(f"  R²:   {test_metrics.r2:.4f}")

        # Create trained model object
        trained_model = TrainedModel(
            name=f"{stat}_model",
            target_stat=stat,
            model=model,
            scaler=scaler,
            features=available_features,
            metrics=test_metrics,
            model_type=algorithm,
            hyperparameters={'alpha': model.alpha if hasattr(model, 'alpha') else None}
        )

        # Store model
        self.shooting_models[stat] = trained_model

        # Log coefficients for linear models
        if hasattr(model, 'coef_'):
            self._log_linear_coefficients(model, available_features, stat)

        return trained_model

    def train_all_models(
        self,
        df: pd.DataFrame,
        counting_algorithm: str = 'gradient_boosting',
        shooting_algorithm: str = 'ridge',
        tune_hyperparams: bool = False
    ) -> Dict[str, TrainedModel]:
        """
        Train all stat models.

        Args:
            df: Training DataFrame
            counting_algorithm: Algorithm for counting stats
            shooting_algorithm: Algorithm for shooting percentages
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Dictionary of all trained models
        """
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ALL FANTASY BASKETBALL ML MODELS")
        logger.info("=" * 60)

        all_models = {}

        # Train counting stat models
        logger.info("\n--- Counting Stats Models ---")
        for stat in COUNTING_STATS:
            if stat in df.columns:
                try:
                    model = self.train_counting_stat_model(
                        df, stat, counting_algorithm, tune_hyperparams
                    )
                    all_models[stat] = model
                except Exception as e:
                    logger.error(f"Failed to train model for {stat}: {e}")
            else:
                logger.warning(f"Stat '{stat}' not found in training data")

        # Train shooting percentage models
        logger.info("\n--- Shooting Percentage Models ---")
        for stat in SHOOTING_STATS:
            if stat in df.columns:
                try:
                    model = self.train_shooting_pct_model(
                        df, stat, shooting_algorithm
                    )
                    all_models[stat] = model
                except Exception as e:
                    logger.error(f"Failed to train model for {stat}: {e}")
            else:
                logger.warning(f"Stat '{stat}' not found in training data")

        return all_models

    def save_all_models(self, prefix: str = '') -> Dict[str, str]:
        """
        Save all trained models to disk.

        Args:
            prefix: Optional prefix for model filenames

        Returns:
            Dictionary mapping stat names to file paths
        """
        saved_paths = {}

        # Save counting models
        for stat, model in self.counting_models.items():
            filename = f"{prefix}{stat}_model.pkl" if prefix else f"{stat}_model.pkl"
            filepath = os.path.join(self.models_dir, filename)
            model.save(filepath)
            saved_paths[stat] = filepath

        # Save shooting models
        for stat, model in self.shooting_models.items():
            filename = f"{prefix}{stat}_model.pkl" if prefix else f"{stat}_model.pkl"
            filepath = os.path.join(self.models_dir, filename)
            model.save(filepath)
            saved_paths[stat] = filepath

        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'models': {
                stat: {
                    'path': path,
                    'metrics': (
                        self.counting_models.get(stat) or
                        self.shooting_models.get(stat)
                    ).metrics.to_dict()
                }
                for stat, path in saved_paths.items()
            },
            'counting_stats': COUNTING_STATS,
            'shooting_stats': SHOOTING_STATS,
        }

        metadata_path = os.path.join(self.models_dir, 'models_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model metadata saved to: {metadata_path}")

        return saved_paths

    def load_all_models(self) -> bool:
        """
        Load all models from disk.

        Returns:
            True if all models loaded successfully
        """
        loaded_count = 0

        for stat in COUNTING_STATS:
            filepath = os.path.join(self.models_dir, f"{stat}_model.pkl")
            if os.path.exists(filepath):
                self.counting_models[stat] = TrainedModel.load(filepath)
                loaded_count += 1
                logger.info(f"Loaded model: {stat}")

        for stat in SHOOTING_STATS:
            filepath = os.path.join(self.models_dir, f"{stat}_model.pkl")
            if os.path.exists(filepath):
                self.shooting_models[stat] = TrainedModel.load(filepath)
                loaded_count += 1
                logger.info(f"Loaded model: {stat}")

        logger.info(f"Loaded {loaded_count} models")
        return loaded_count > 0

    def predict_player_stats(
        self,
        player_features: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Predict all stats for given player features.

        Args:
            player_features: DataFrame with player features

        Returns:
            Dictionary mapping stat names to predictions
        """
        predictions = {}

        # Counting stats
        for stat, model in self.counting_models.items():
            try:
                predictions[stat] = model.predict(player_features)
            except Exception as e:
                logger.warning(f"Could not predict {stat}: {e}")
                predictions[stat] = np.zeros(len(player_features))

        # Shooting percentages
        for stat, model in self.shooting_models.items():
            try:
                preds = model.predict(player_features)
                predictions[stat] = np.clip(preds, 0, 1)  # Ensure valid range
            except Exception as e:
                logger.warning(f"Could not predict {stat}: {e}")
                predictions[stat] = np.zeros(len(player_features))

        return predictions

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> ModelMetrics:
        """Calculate evaluation metrics for predictions."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)

        # Cross-validation RMSE
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring='neg_root_mean_squared_error'
        )
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()

        return ModelMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            explained_variance=explained_var,
            cv_rmse_mean=cv_rmse_mean,
            cv_rmse_std=cv_rmse_std
        )

    def _log_feature_importance(
        self,
        model: Any,
        features: List[str],
        stat: str,
        algorithm: str
    ) -> None:
        """Log feature importances for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            logger.info(f"\nTop 10 Feature Importances for {stat.upper()}:")
            for i in range(min(10, len(features))):
                idx = indices[i]
                logger.info(f"  {i+1}. {features[idx]}: {importances[idx]:.4f}")

    def _log_linear_coefficients(
        self,
        model: Any,
        features: List[str],
        stat: str
    ) -> None:
        """Log coefficients for linear models."""
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            indices = np.argsort(np.abs(coefs))[::-1]

            logger.info(f"\nTop 10 Coefficients for {stat.upper()}:")
            for i in range(min(10, len(features))):
                idx = indices[i]
                logger.info(f"  {i+1}. {features[idx]}: {coefs[idx]:.4f}")


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_model_suite(
    models: Dict[str, TrainedModel],
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate all models on a test dataset.

    Args:
        models: Dictionary of trained models
        test_df: Test DataFrame

    Returns:
        DataFrame with evaluation results
    """
    results = []

    for stat, model in models.items():
        try:
            X, y = prepare_features(test_df, model.features, stat)
            y_pred = model.predict(X)

            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            results.append({
                'stat': stat,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_samples': len(y)
            })
        except Exception as e:
            logger.error(f"Error evaluating {stat}: {e}")

    return pd.DataFrame(results)


def generate_model_report(ml_model: FantasyBasketballMLModel) -> str:
    """
    Generate a summary report of all trained models.

    Args:
        ml_model: FantasyBasketballMLModel instance

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "FANTASY BASKETBALL ML MODEL REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "COUNTING STATS MODELS",
        "-" * 40
    ]

    for stat, model in ml_model.counting_models.items():
        lines.extend([
            f"\n{stat.upper()}:",
            f"  Algorithm: {model.model_type}",
            f"  Features:  {len(model.features)}",
            f"  RMSE:      {model.metrics.rmse:.4f}",
            f"  MAE:       {model.metrics.mae:.4f}",
            f"  R²:        {model.metrics.r2:.4f}",
        ])

    lines.extend([
        "",
        "SHOOTING PERCENTAGE MODELS",
        "-" * 40
    ])

    for stat, model in ml_model.shooting_models.items():
        lines.extend([
            f"\n{stat.upper()}:",
            f"  Algorithm: {model.model_type}",
            f"  Features:  {len(model.features)}",
            f"  RMSE:      {model.metrics.rmse:.4f}",
            f"  MAE:       {model.metrics.mae:.4f}",
            f"  R²:        {model.metrics.r2:.4f}",
        ])

    lines.extend([
        "",
        "=" * 60
    ])

    return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for model training."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Fantasy Basketball ML Models'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=DEFAULT_TRAINING_DATA,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--counting-algo',
        type=str,
        choices=['gradient_boosting', 'random_forest'],
        default='gradient_boosting',
        help='Algorithm for counting stats'
    )
    parser.add_argument(
        '--shooting-algo',
        type=str,
        choices=['ridge', 'lasso', 'elastic_net'],
        default='ridge',
        help='Algorithm for shooting percentages'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (slower)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=MODELS_DIR,
        help='Directory to save trained models'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Fantasy Basketball ML Model Training")
    logger.info("=" * 60)
    logger.info(f"Training data: {args.data}")
    logger.info(f"Counting algorithm: {args.counting_algo}")
    logger.info(f"Shooting algorithm: {args.shooting_algo}")
    logger.info(f"Hyperparameter tuning: {args.tune}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    try:
        # Initialize model manager
        ml_model = FantasyBasketballMLModel(models_dir=args.output_dir)

        # Load training data
        df = ml_model.load_training_data(args.data)

        # Train all models
        models = ml_model.train_all_models(
            df,
            counting_algorithm=args.counting_algo,
            shooting_algorithm=args.shooting_algo,
            tune_hyperparams=args.tune
        )

        # Save models
        saved_paths = ml_model.save_all_models()

        # Generate report
        report = generate_model_report(ml_model)
        print("\n" + report)

        # Save report
        report_path = os.path.join(args.output_dir, 'training_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Models trained: {len(models)}")
        logger.info(f"Saved to: {args.output_dir}")
        for stat, path in saved_paths.items():
            logger.info(f"  - {stat}: {os.path.basename(path)}")
        logger.info("=" * 60)

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

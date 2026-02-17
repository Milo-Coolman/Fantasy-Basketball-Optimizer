#!/usr/bin/env python3
"""
Training Data Collection Script for Fantasy Basketball Projections.

This script collects historical NBA player statistics from Basketball Reference
for the 2019-2024 seasons and processes them into features suitable for
machine learning model training.

Features Generated:
- Raw per-game statistics
- Per-36 minute statistics
- Advanced stats (PER, USG%, TS%, etc.)
- Derived metrics (fantasy points, efficiency ratings)
- Season context (games into season, team pace)

Usage:
    python backend/projections/collect_training_data.py [--seasons 2019-2024] [--output training_data.csv]

Output:
    CSV file in backend/projections/training_data/
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.scrapers.basketball_reference import (
    BasketballReferenceScraper,
    ScraperError,
    PageNotFoundError,
    normalize_player_name,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_data_collection.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default seasons to collect (2019 = 2018-19 season through 2024 = 2023-24)
DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'training_data'
)

# Minimum games played threshold to include a player-season
MIN_GAMES_PLAYED = 10

# Minimum minutes per game to avoid small sample issues
MIN_MINUTES_PER_GAME = 10.0

# Fantasy scoring weights (standard ESPN H2H categories)
FANTASY_WEIGHTS = {
    'pts': 1.0,
    'trb': 1.2,
    'ast': 1.5,
    'stl': 3.0,
    'blk': 3.0,
    'tov': -1.0,
    'fg_pct': 0.0,  # Handled separately as category
    'ft_pct': 0.0,  # Handled separately as category
    '3p': 1.0,
}

# Columns to keep from per-game stats
PER_GAME_COLUMNS = [
    'player', 'player_id', 'pos', 'age', 'tm', 'g', 'gs', 'mp',
    'fg', 'fga', 'fg_pct', '3p', '3pa', '3p_pct',
    '2p', '2pa', '2p_pct', 'efg_pct', 'ft', 'fta', 'ft_pct',
    'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts',
    'season', 'player_normalized'
]

# Columns to keep from advanced stats
ADVANCED_COLUMNS = [
    'player', 'tm', 'per', 'ts_pct', '3par', 'ftr',
    'orb_pct', 'drb_pct', 'trb_pct', 'ast_pct', 'stl_pct', 'blk_pct',
    'tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_48',
    'obpm', 'dbpm', 'bpm', 'vorp'
]


# =============================================================================
# Data Collection
# =============================================================================

def collect_season_data(
    scraper: BasketballReferenceScraper,
    season: int
) -> Optional[pd.DataFrame]:
    """
    Collect all player statistics for a single season.

    Args:
        scraper: Basketball Reference scraper instance
        season: Season year to collect

    Returns:
        DataFrame with combined stats or None if failed
    """
    logger.info(f"Collecting data for {season-1}-{str(season)[2:]} season...")

    try:
        # Get per-game stats
        logger.info(f"  Fetching per-game stats...")
        per_game = scraper.get_per_game_stats(season)

        # Get advanced stats
        logger.info(f"  Fetching advanced stats...")
        advanced = scraper.get_advanced_stats(season)

        # Get totals for calculating per-36 stats
        logger.info(f"  Fetching totals...")
        totals = scraper.get_totals_stats(season)

        # Filter columns
        per_game_cols = [c for c in PER_GAME_COLUMNS if c in per_game.columns]
        per_game = per_game[per_game_cols].copy()

        advanced_cols = [c for c in ADVANCED_COLUMNS if c in advanced.columns]
        advanced = advanced[advanced_cols].copy()

        # Merge per-game and advanced stats
        df = per_game.merge(
            advanced,
            on=['player', 'tm'],
            how='left',
            suffixes=('', '_adv')
        )

        # Add totals for per-36 calculation
        totals_cols = ['player', 'tm', 'mp', 'pts', 'trb', 'ast', 'stl', 'blk',
                       'fg', 'fga', 'ft', 'fta', '3p', '3pa', 'tov', 'pf', 'orb', 'drb']
        totals_available = [c for c in totals_cols if c in totals.columns]
        totals_subset = totals[totals_available].copy()
        totals_subset = totals_subset.rename(columns={
            c: f'{c}_total' for c in totals_subset.columns if c not in ['player', 'tm']
        })

        df = df.merge(
            totals_subset,
            on=['player', 'tm'],
            how='left'
        )

        logger.info(f"  Collected {len(df)} player-seasons")
        return df

    except PageNotFoundError:
        logger.warning(f"Season {season} not available on Basketball Reference")
        return None
    except ScraperError as e:
        logger.error(f"Error collecting data for {season}: {e}")
        return None


def collect_all_seasons(
    seasons: List[int],
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Collect data for multiple seasons and combine.

    Args:
        seasons: List of season years to collect
        output_file: Optional intermediate output file

    Returns:
        Combined DataFrame with all seasons
    """
    scraper = BasketballReferenceScraper()
    all_data = []

    for season in seasons:
        df = collect_season_data(scraper, season)
        if df is not None:
            all_data.append(df)

            # Save intermediate progress
            if output_file:
                pd.concat(all_data, ignore_index=True).to_csv(
                    output_file.replace('.csv', '_intermediate.csv'),
                    index=False
                )

    if not all_data:
        raise ValueError("No data collected for any season")

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total records collected: {len(combined)}")

    return combined


# =============================================================================
# Feature Engineering
# =============================================================================

def calculate_per_36_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-36 minute statistics.

    Per-36 stats normalize production to a standard 36 minutes,
    allowing comparison across players with different playing times.

    Args:
        df: DataFrame with total stats columns

    Returns:
        DataFrame with per-36 columns added
    """
    logger.info("Calculating per-36 minute stats...")

    # Stats to convert to per-36
    counting_stats = ['pts', 'trb', 'ast', 'stl', 'blk', 'fg', 'fga',
                      'ft', 'fta', '3p', '3pa', 'tov', 'pf', 'orb', 'drb']

    for stat in counting_stats:
        total_col = f'{stat}_total'
        mp_col = 'mp_total'

        if total_col in df.columns and mp_col in df.columns:
            # Per-36 = (stat_total / minutes_total) * 36
            df[f'{stat}_per36'] = np.where(
                df[mp_col] > 0,
                (df[total_col] / df[mp_col]) * 36,
                0
            )

    return df


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features useful for ML models.

    Features include:
    - Fantasy point estimates
    - Efficiency metrics
    - Shooting splits
    - Usage-adjusted stats

    Args:
        df: DataFrame with raw stats

    Returns:
        DataFrame with derived features added
    """
    logger.info("Calculating derived features...")

    # Fantasy points estimate (simplified scoring)
    df['fantasy_pts'] = (
        df['pts'] * FANTASY_WEIGHTS['pts'] +
        df['trb'] * FANTASY_WEIGHTS['trb'] +
        df['ast'] * FANTASY_WEIGHTS['ast'] +
        df['stl'] * FANTASY_WEIGHTS['stl'] +
        df['blk'] * FANTASY_WEIGHTS['blk'] +
        df['tov'] * FANTASY_WEIGHTS['tov'] +
        df['3p'] * FANTASY_WEIGHTS['3p']
    )

    # Double-double potential (simple heuristic)
    df['dd_potential'] = (
        (df['pts'] >= 10).astype(int) +
        (df['trb'] >= 10).astype(int) +
        (df['ast'] >= 10).astype(int)
    )

    # Scoring efficiency (points per shot attempt)
    df['scoring_efficiency'] = np.where(
        df['fga'] > 0,
        df['pts'] / df['fga'],
        0
    )

    # Free throw rate (FTA per FGA)
    df['ft_rate'] = np.where(
        df['fga'] > 0,
        df['fta'] / df['fga'],
        0
    )

    # Three point rate (3PA per FGA)
    df['3p_rate'] = np.where(
        df['fga'] > 0,
        df['3pa'] / df['fga'],
        0
    )

    # Assist to turnover ratio
    df['ast_tov_ratio'] = np.where(
        df['tov'] > 0,
        df['ast'] / df['tov'],
        df['ast']  # If no turnovers, use assists
    )

    # Stock (steals + blocks)
    df['stocks'] = df['stl'] + df['blk']

    # Rebound percentage (if available)
    if 'trb_pct' not in df.columns and 'orb_pct' in df.columns and 'drb_pct' in df.columns:
        df['trb_pct'] = (df['orb_pct'] + df['drb_pct']) / 2

    # Games started percentage
    df['gs_pct'] = np.where(
        df['g'] > 0,
        df['gs'] / df['g'],
        0
    )

    # Minutes per game (if not already present)
    if 'mpg' not in df.columns:
        df['mpg'] = df['mp']

    # Experience proxy (age bracket)
    df['age_bracket'] = pd.cut(
        df['age'],
        bins=[0, 22, 25, 28, 32, 100],
        labels=['rookie', 'young', 'prime', 'veteran', 'late_career']
    )

    # Position encoding (simplified)
    df['is_guard'] = df['pos'].str.contains('G', na=False).astype(int)
    df['is_forward'] = df['pos'].str.contains('F', na=False).astype(int)
    df['is_center'] = df['pos'].str.contains('C', na=False).astype(int)

    return df


def calculate_year_over_year_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year changes for tracking player development.

    Args:
        df: DataFrame with multiple seasons per player

    Returns:
        DataFrame with YoY change columns added
    """
    logger.info("Calculating year-over-year changes...")

    # Sort by player and season
    df = df.sort_values(['player_normalized', 'season'])

    # Stats to track changes
    change_stats = ['pts', 'trb', 'ast', 'stl', 'blk', 'fg_pct', 'ft_pct',
                    '3p_pct', 'mp', 'usg_pct', 'ts_pct', 'fantasy_pts']

    for stat in change_stats:
        if stat in df.columns:
            # Calculate change from previous season
            df[f'{stat}_yoy_change'] = df.groupby('player_normalized')[stat].diff()

            # Calculate percentage change
            df[f'{stat}_yoy_pct_change'] = df.groupby('player_normalized')[stat].pct_change()

    # Previous season values (for comparison)
    for stat in ['pts', 'trb', 'ast', 'mp', 'g']:
        if stat in df.columns:
            df[f'{stat}_prev_season'] = df.groupby('player_normalized')[stat].shift(1)

    return df


# =============================================================================
# Data Cleaning
# =============================================================================

def clean_data(
    df: pd.DataFrame,
    min_games: int = MIN_GAMES_PLAYED,
    min_minutes: float = MIN_MINUTES_PER_GAME
) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers.

    Args:
        df: Raw DataFrame
        min_games: Minimum games played threshold
        min_minutes: Minimum minutes per game threshold

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    initial_rows = len(df)

    # 1. Remove rows with missing player IDs or names
    df = df.dropna(subset=['player'])
    logger.info(f"  After removing missing players: {len(df)} rows")

    # 2. Filter by minimum games played
    df = df[df['g'] >= min_games]
    logger.info(f"  After min games filter ({min_games}+): {len(df)} rows")

    # 3. Filter by minimum minutes per game
    df = df[df['mp'] >= min_minutes]
    logger.info(f"  After min minutes filter ({min_minutes}+): {len(df)} rows")

    # 4. Handle missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Fill missing percentages with 0 (player didn't attempt those shots)
    pct_cols = [c for c in numeric_cols if 'pct' in c.lower()]
    df[pct_cols] = df[pct_cols].fillna(0)

    # Fill missing counting stats with 0
    counting_cols = ['fg', 'fga', '3p', '3pa', 'ft', 'fta', 'orb', 'drb',
                     'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']
    for col in counting_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill missing advanced stats with league average or 0
    advanced_cols = ['per', 'ts_pct', 'usg_pct', 'ws', 'bpm', 'vorp']
    for col in advanced_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    # 5. Handle outliers using IQR method for key stats
    outlier_cols = ['pts', 'trb', 'ast', 'mp', 'fantasy_pts']
    for col in outlier_cols:
        if col in df.columns:
            df = handle_outliers(df, col)

    # 6. Ensure percentages are in valid range [0, 1]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)

    # 7. Remove duplicate player-seasons (keep first occurrence)
    df = df.drop_duplicates(subset=['player_normalized', 'season'], keep='first')

    logger.info(f"  Final cleaned dataset: {len(df)} rows (removed {initial_rows - len(df)})")

    return df


def handle_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    factor: float = 3.0
) -> pd.DataFrame:
    """
    Handle outliers in a specific column.

    Args:
        df: DataFrame
        column: Column to check for outliers
        method: 'iqr' for interquartile range, 'zscore' for z-score
        factor: IQR multiplier or z-score threshold

    Returns:
        DataFrame with outliers handled
    """
    if column not in df.columns:
        return df

    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Cap outliers instead of removing (preserve data)
        df[column] = df[column].clip(lower_bound, upper_bound)

    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()

        if std > 0:
            z_scores = (df[column] - mean) / std
            df = df[abs(z_scores) <= factor]

    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the cleaned dataset.

    Args:
        df: Cleaned DataFrame

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check for required columns
    required_cols = ['player', 'season', 'g', 'mp', 'pts', 'trb', 'ast']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check for empty dataframe
    if len(df) == 0:
        issues.append("DataFrame is empty")

    # Check for reasonable value ranges
    if 'pts' in df.columns and df['pts'].max() > 50:
        issues.append(f"Suspicious max points per game: {df['pts'].max()}")

    if 'mp' in df.columns and df['mp'].max() > 48:
        issues.append(f"Suspicious max minutes per game: {df['mp'].max()}")

    # Check for seasons coverage
    if 'season' in df.columns:
        seasons = df['season'].unique()
        logger.info(f"  Seasons in dataset: {sorted(seasons)}")

    # Check for duplicate player-seasons
    if 'player_normalized' in df.columns and 'season' in df.columns:
        duplicates = df.duplicated(subset=['player_normalized', 'season'], keep=False)
        if duplicates.any():
            issues.append(f"Found {duplicates.sum()} duplicate player-seasons")

    is_valid = len(issues) == 0
    return is_valid, issues


# =============================================================================
# Output
# =============================================================================

def save_training_data(
    df: pd.DataFrame,
    filename: str,
    include_metadata: bool = True
) -> str:
    """
    Save the training data to CSV.

    Args:
        df: Processed DataFrame
        filename: Output filename
        include_metadata: Whether to save a metadata file

    Returns:
        Path to saved file
    """
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full output path
    if not filename.endswith('.csv'):
        filename += '.csv'
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Save main data
    df.to_csv(output_path, index=False)
    logger.info(f"Saved training data to: {output_path}")

    # Save metadata
    if include_metadata:
        metadata = {
            'created_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': len(df.columns),
            'seasons': sorted(df['season'].unique().tolist()) if 'season' in df.columns else [],
            'unique_players': df['player'].nunique() if 'player' in df.columns else 0,
            'features': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
        }

        metadata_path = output_path.replace('.csv', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_path}")

    return output_path


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the training data.

    Args:
        df: Training DataFrame

    Returns:
        Summary statistics DataFrame
    """
    # Numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    summary = numeric_df.describe()

    # Add additional stats
    summary.loc['missing'] = numeric_df.isnull().sum()
    summary.loc['missing_pct'] = (numeric_df.isnull().sum() / len(df)) * 100

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for data collection."""
    parser = argparse.ArgumentParser(
        description='Collect NBA training data from Basketball Reference'
    )
    parser.add_argument(
        '--seasons',
        type=str,
        default='2019-2024',
        help='Season range to collect (e.g., "2019-2024" or "2020,2021,2022")'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='nba_training_data.csv',
        help='Output filename'
    )
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='Skip data cleaning step'
    )
    parser.add_argument(
        '--min-games',
        type=int,
        default=MIN_GAMES_PLAYED,
        help=f'Minimum games played (default: {MIN_GAMES_PLAYED})'
    )
    parser.add_argument(
        '--min-minutes',
        type=float,
        default=MIN_MINUTES_PER_GAME,
        help=f'Minimum minutes per game (default: {MIN_MINUTES_PER_GAME})'
    )

    args = parser.parse_args()

    # Parse seasons
    if '-' in args.seasons:
        start, end = args.seasons.split('-')
        seasons = list(range(int(start), int(end) + 1))
    else:
        seasons = [int(s.strip()) for s in args.seasons.split(',')]

    # Update thresholds based on args
    min_games = args.min_games
    min_minutes = args.min_minutes

    logger.info("=" * 60)
    logger.info("Fantasy Basketball Training Data Collection")
    logger.info("=" * 60)
    logger.info(f"Seasons to collect: {seasons}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Min games: {min_games}, Min minutes: {min_minutes}")
    logger.info("=" * 60)

    try:
        # Step 1: Collect data from all seasons
        logger.info("\nStep 1: Collecting data from Basketball Reference...")
        df = collect_all_seasons(seasons, args.output)

        # Step 2: Calculate per-36 stats
        logger.info("\nStep 2: Calculating per-36 minute stats...")
        df = calculate_per_36_stats(df)

        # Step 3: Calculate derived features
        logger.info("\nStep 3: Calculating derived features...")
        df = calculate_derived_features(df)

        # Step 4: Calculate year-over-year changes
        logger.info("\nStep 4: Calculating year-over-year changes...")
        df = calculate_year_over_year_changes(df)

        # Step 5: Clean data (unless skipped)
        if not args.skip_cleaning:
            logger.info("\nStep 5: Cleaning data...")
            df = clean_data(df, min_games=min_games, min_minutes=min_minutes)

        # Step 6: Validate data
        logger.info("\nStep 6: Validating data...")
        is_valid, issues = validate_data(df)
        if not is_valid:
            logger.warning(f"Validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("  Data validation passed!")

        # Step 7: Save data
        logger.info("\nStep 7: Saving training data...")
        output_path = save_training_data(df, args.output)

        # Step 8: Generate and save summary
        logger.info("\nStep 8: Generating summary statistics...")
        summary = generate_summary_statistics(df)
        summary_path = output_path.replace('.csv', '_summary.csv')
        summary.to_csv(summary_path)
        logger.info(f"Saved summary to: {summary_path}")

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("Collection Complete!")
        logger.info("=" * 60)
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Unique players: {df['player'].nunique()}")
        logger.info(f"Seasons: {sorted(df['season'].unique())}")
        logger.info(f"Features: {len(df.columns)}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nCollection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nError during collection: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

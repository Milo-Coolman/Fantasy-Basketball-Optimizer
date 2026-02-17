#!/usr/bin/env python3
"""
Rotisserie (Roto) League Analyzer for Fantasy Basketball.

This module provides analysis specifically for Rotisserie leagues, including:
- End-of-season category total projections
- Category ranking predictions
- Overall league win probability
- Category strength/weakness identification
- Gap analysis to next rank
- Punt strategy recommendations

In Roto leagues, teams accumulate stats throughout the season and are
ranked 1st-12th (or league size) in each category. Points are awarded
based on rank (12 points for 1st, 1 point for 12th), and the team with
the most total points wins.

Reference: PRD Section 3.3.6 - Roto League Projections
"""

import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
from scipy import stats as scipy_stats

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

# Standard Roto categories (9-cat)
STANDARD_CATEGORIES = [
    'pts', 'trb', 'ast', 'stl', 'blk', '3p', 'fg_pct', 'ft_pct', 'tov'
]

# Categories where lower is better
NEGATIVE_CATEGORIES = ['tov']

# Counting stats (accumulate over season)
COUNTING_CATEGORIES = ['pts', 'trb', 'ast', 'stl', 'blk', '3p', 'tov']

# Rate stats (calculated from totals)
RATE_CATEGORIES = ['fg_pct', 'ft_pct']

# Default league size
DEFAULT_LEAGUE_SIZE = 12

# Default games remaining estimate
DEFAULT_GAMES_REMAINING = 40  # Per player

# Simulation settings
DEFAULT_SIMULATIONS = 10000

# Variance factors for projections
PROJECTION_VARIANCE = {
    'pts': 0.08,
    'trb': 0.10,
    'ast': 0.12,
    'stl': 0.18,
    'blk': 0.20,
    '3p': 0.15,
    'fg_pct': 0.03,
    'ft_pct': 0.04,
    'tov': 0.12,
}


# =============================================================================
# Data Classes
# =============================================================================

class CategoryStrength(Enum):
    """Category strength classification."""
    DOMINANT = "dominant"      # Top 2
    STRONG = "strong"          # 3-4
    AVERAGE = "average"        # 5-8
    WEAK = "weak"              # 9-10
    PUNT = "punt"              # Bottom 2


@dataclass
class TeamRotoStats:
    """Roto statistics for a team."""
    team_id: int
    team_name: str

    # Current season totals
    current_totals: Dict[str, float]

    # For rate stats, need makes and attempts
    fgm: float = 0
    fga: float = 0
    ftm: float = 0
    fta: float = 0

    # Projected rest-of-season per-game averages
    ros_projections: Dict[str, float] = field(default_factory=dict)

    # Games remaining
    games_remaining: int = DEFAULT_GAMES_REMAINING
    roster_size: int = 10

    def get_projected_totals(self) -> Dict[str, float]:
        """Calculate projected end-of-season totals."""
        totals = {}

        for cat in COUNTING_CATEGORIES:
            current = self.current_totals.get(cat, 0)
            ros_per_game = self.ros_projections.get(cat, 0)
            ros_total = ros_per_game * self.games_remaining * self.roster_size
            totals[cat] = current + ros_total

        # Rate stats need special handling
        # Project additional makes/attempts
        ros_fgm = self.ros_projections.get('fgm', 0) * self.games_remaining * self.roster_size
        ros_fga = self.ros_projections.get('fga', 0) * self.games_remaining * self.roster_size
        ros_ftm = self.ros_projections.get('ftm', 0) * self.games_remaining * self.roster_size
        ros_fta = self.ros_projections.get('fta', 0) * self.games_remaining * self.roster_size

        total_fgm = self.fgm + ros_fgm
        total_fga = self.fga + ros_fga
        total_ftm = self.ftm + ros_ftm
        total_fta = self.fta + ros_fta

        totals['fg_pct'] = total_fgm / max(total_fga, 1)
        totals['ft_pct'] = total_ftm / max(total_fta, 1)

        return totals


@dataclass
class CategoryRanking:
    """Ranking information for a single category."""
    category: str
    current_rank: int
    projected_rank: float
    rank_probability: Dict[int, float]  # rank -> probability

    current_value: float
    projected_value: float

    # Gap analysis
    gap_to_next: float  # Positive = ahead, negative = behind
    gap_to_previous: float
    next_team: Optional[str] = None
    previous_team: Optional[str] = None

    # Strength classification
    strength: CategoryStrength = CategoryStrength.AVERAGE

    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category,
            'current_rank': self.current_rank,
            'projected_rank': round(self.projected_rank, 1),
            'current_value': round(self.current_value, 2),
            'projected_value': round(self.projected_value, 2),
            'gap_to_next': round(self.gap_to_next, 2),
            'strength': self.strength.value,
            'next_team': self.next_team,
        }


@dataclass
class RotoProjection:
    """Complete Roto projection for a team."""
    team_id: int
    team_name: str

    # Current standing
    current_points: float
    current_rank: int

    # Projected standing
    projected_points: float
    projected_rank: float
    rank_distribution: Dict[int, float]

    # Win probability
    win_probability: float
    top_3_probability: float
    bottom_3_probability: float

    # Category analysis
    category_rankings: Dict[str, CategoryRanking]
    strengths: List[str]
    weaknesses: List[str]

    # Strategy recommendations
    punt_candidates: List[str]
    improvement_targets: List[Dict[str, Any]]

    # Metadata
    games_remaining: int
    projection_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'team_id': self.team_id,
            'team_name': self.team_name,
            'current_points': round(self.current_points, 1),
            'current_rank': self.current_rank,
            'projected_points': round(self.projected_points, 1),
            'projected_rank': round(self.projected_rank, 1),
            'win_probability': round(self.win_probability * 100, 1),
            'top_3_probability': round(self.top_3_probability * 100, 1),
            'category_rankings': {k: v.to_dict() for k, v in self.category_rankings.items()},
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'punt_candidates': self.punt_candidates,
            'improvement_targets': self.improvement_targets,
            'games_remaining': self.games_remaining,
        }


@dataclass
class GapAnalysis:
    """Gap analysis for moving up/down in category rankings."""
    category: str
    current_rank: int
    current_value: float

    # Gap to move up one rank
    gap_to_improve: float
    team_ahead: Optional[str]
    team_ahead_value: float

    # Cushion before dropping one rank
    cushion: float
    team_behind: Optional[str]
    team_behind_value: float

    # Daily rate needed to move up
    daily_rate_to_improve: float
    games_needed_to_improve: int

    # Is this achievable?
    improvement_difficulty: str  # 'easy', 'moderate', 'hard', 'unlikely'


@dataclass
class PuntStrategy:
    """Punt strategy recommendation."""
    punt_categories: List[str]
    points_sacrificed: float
    resources_freed: str  # Description of what you can gain

    # Impact analysis
    current_points: float
    projected_points_with_punt: float
    ranking_impact: Dict[str, int]  # category -> rank change

    # Recommendation strength
    viability_score: float  # 0-100
    recommendation: str


# =============================================================================
# Roto Analyzer
# =============================================================================

class RotoAnalyzer:
    """
    Analyzer for Rotisserie fantasy basketball leagues.

    Provides end-of-season projections, category rankings, win probabilities,
    and strategic recommendations for Roto leagues.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        negative_categories: Optional[List[str]] = None,
        league_size: int = DEFAULT_LEAGUE_SIZE,
        num_simulations: int = DEFAULT_SIMULATIONS
    ):
        """
        Initialize the Roto analyzer.

        Args:
            categories: Stat categories (default: 9-cat)
            negative_categories: Categories where lower is better
            league_size: Number of teams in the league
            num_simulations: Number of Monte Carlo iterations
        """
        self.categories = categories or STANDARD_CATEGORIES.copy()
        self.negative_categories = set(negative_categories or NEGATIVE_CATEGORIES)
        self.league_size = league_size
        self.num_simulations = num_simulations

        # Initialize random generator
        self.rng = np.random.default_rng()

    # =========================================================================
    # Main Analysis Methods
    # =========================================================================

    def analyze_team(
        self,
        team: TeamRotoStats,
        all_teams: List[TeamRotoStats],
        games_remaining: Optional[int] = None
    ) -> RotoProjection:
        """
        Generate complete Roto analysis for a team.

        Args:
            team: Team to analyze
            all_teams: All teams in the league
            games_remaining: Override games remaining estimate

        Returns:
            RotoProjection with full analysis
        """
        if games_remaining is not None:
            team.games_remaining = games_remaining

        logger.debug(f"Analyzing Roto standings for: {team.team_name}")

        # Calculate current standings
        current_rankings = self._calculate_current_rankings(all_teams)
        current_points = self._calculate_roto_points(team.team_id, current_rankings)
        current_rank = self._get_overall_rank(team.team_id, all_teams, current_rankings)

        # Project end-of-season totals
        projected_totals = {t.team_id: t.get_projected_totals() for t in all_teams}

        # Simulate season end
        sim_results = self._simulate_season(all_teams, projected_totals)

        # Calculate projected standings
        projected_rankings = self._calculate_projected_rankings(projected_totals)
        projected_points = np.mean(sim_results['points'][team.team_id])
        projected_rank = np.mean(sim_results['ranks'][team.team_id])
        rank_distribution = self._calculate_rank_distribution(sim_results['ranks'][team.team_id])

        # Win probability
        win_prob = sim_results['wins'][team.team_id] / self.num_simulations
        top_3_prob = sum(1 for r in sim_results['ranks'][team.team_id] if r <= 3) / self.num_simulations
        bottom_3_prob = sum(1 for r in sim_results['ranks'][team.team_id] if r > self.league_size - 3) / self.num_simulations

        # Category analysis
        category_rankings = self._analyze_categories(
            team, all_teams, current_rankings, projected_rankings, projected_totals
        )

        # Identify strengths and weaknesses
        strengths = [cat for cat, ranking in category_rankings.items()
                     if ranking.strength in [CategoryStrength.DOMINANT, CategoryStrength.STRONG]]
        weaknesses = [cat for cat, ranking in category_rankings.items()
                      if ranking.strength in [CategoryStrength.WEAK, CategoryStrength.PUNT]]

        # Strategy recommendations
        punt_candidates = self._identify_punt_candidates(category_rankings, projected_points)
        improvement_targets = self._identify_improvement_targets(
            team, all_teams, category_rankings, projected_totals
        )

        return RotoProjection(
            team_id=team.team_id,
            team_name=team.team_name,
            current_points=current_points,
            current_rank=current_rank,
            projected_points=projected_points,
            projected_rank=projected_rank,
            rank_distribution=rank_distribution,
            win_probability=win_prob,
            top_3_probability=top_3_prob,
            bottom_3_probability=bottom_3_prob,
            category_rankings=category_rankings,
            strengths=strengths,
            weaknesses=weaknesses,
            punt_candidates=punt_candidates,
            improvement_targets=improvement_targets,
            games_remaining=team.games_remaining
        )

    def analyze_league(
        self,
        all_teams: List[TeamRotoStats],
        games_remaining: Optional[int] = None
    ) -> List[RotoProjection]:
        """
        Analyze all teams in the league.

        Args:
            all_teams: All teams in the league
            games_remaining: Override games remaining estimate

        Returns:
            List of RotoProjection sorted by projected rank
        """
        projections = []

        for team in all_teams:
            proj = self.analyze_team(team, all_teams, games_remaining)
            projections.append(proj)

        # Sort by projected rank
        projections.sort(key=lambda p: p.projected_rank)

        return projections

    def get_gap_analysis(
        self,
        team: TeamRotoStats,
        all_teams: List[TeamRotoStats]
    ) -> Dict[str, GapAnalysis]:
        """
        Get detailed gap analysis for each category.

        Args:
            team: Team to analyze
            all_teams: All teams in the league

        Returns:
            Dictionary of category -> GapAnalysis
        """
        # Calculate current and projected standings
        current_rankings = self._calculate_current_rankings(all_teams)
        projected_totals = {t.team_id: t.get_projected_totals() for t in all_teams}
        projected_rankings = self._calculate_projected_rankings(projected_totals)

        team_lookup = {t.team_id: t for t in all_teams}
        gap_analysis = {}

        for cat in self.categories:
            current_rank = current_rankings[cat].index(team.team_id) + 1
            current_value = team.current_totals.get(cat, 0)
            projected_value = projected_totals[team.team_id].get(cat, 0)

            # Find teams ahead and behind
            cat_ranking = current_rankings[cat]
            my_idx = cat_ranking.index(team.team_id)

            # Gap to team ahead (if not first)
            if my_idx > 0:
                ahead_id = cat_ranking[my_idx - 1]
                ahead_team = team_lookup[ahead_id]
                ahead_value = ahead_team.current_totals.get(cat, 0)
                team_ahead = ahead_team.team_name

                if cat in self.negative_categories:
                    gap_to_improve = current_value - ahead_value  # Need to be lower
                else:
                    gap_to_improve = ahead_value - current_value  # Need to be higher
            else:
                gap_to_improve = 0
                team_ahead = None
                ahead_value = current_value

            # Cushion to team behind (if not last)
            if my_idx < len(cat_ranking) - 1:
                behind_id = cat_ranking[my_idx + 1]
                behind_team = team_lookup[behind_id]
                behind_value = behind_team.current_totals.get(cat, 0)
                team_behind = behind_team.team_name

                if cat in self.negative_categories:
                    cushion = behind_value - current_value  # They need to be lower
                else:
                    cushion = current_value - behind_value  # They need to be higher
            else:
                cushion = float('inf')
                team_behind = None
                behind_value = current_value

            # Calculate daily rate needed
            if gap_to_improve > 0 and team.games_remaining > 0:
                daily_rate = gap_to_improve / (team.games_remaining * team.roster_size)
                games_needed = int(gap_to_improve / max(team.ros_projections.get(cat, 0.1), 0.1))
            else:
                daily_rate = 0
                games_needed = 0

            # Difficulty assessment
            if gap_to_improve <= 0:
                difficulty = 'easy'  # Already ahead or tied
            elif games_needed <= team.games_remaining * 0.3:
                difficulty = 'easy'
            elif games_needed <= team.games_remaining * 0.7:
                difficulty = 'moderate'
            elif games_needed <= team.games_remaining:
                difficulty = 'hard'
            else:
                difficulty = 'unlikely'

            gap_analysis[cat] = GapAnalysis(
                category=cat,
                current_rank=current_rank,
                current_value=current_value,
                gap_to_improve=gap_to_improve,
                team_ahead=team_ahead,
                team_ahead_value=ahead_value,
                cushion=cushion,
                team_behind=team_behind,
                team_behind_value=behind_value,
                daily_rate_to_improve=daily_rate,
                games_needed_to_improve=games_needed,
                improvement_difficulty=difficulty
            )

        return gap_analysis

    def evaluate_punt_strategy(
        self,
        team: TeamRotoStats,
        all_teams: List[TeamRotoStats],
        punt_categories: List[str]
    ) -> PuntStrategy:
        """
        Evaluate a specific punt strategy.

        Args:
            team: Team to analyze
            all_teams: All teams in the league
            punt_categories: Categories to punt

        Returns:
            PuntStrategy evaluation
        """
        # Current projected standings
        projected_totals = {t.team_id: t.get_projected_totals() for t in all_teams}
        current_rankings = self._calculate_projected_rankings(projected_totals)

        # Calculate current projected points
        current_points = 0
        for cat in self.categories:
            rank = current_rankings[cat].index(team.team_id) + 1
            current_points += (self.league_size + 1 - rank)

        # Calculate points if punting (assume last place in punt categories)
        punt_points = 0
        ranking_impact = {}

        for cat in self.categories:
            if cat in punt_categories:
                # Assume last place
                punt_points += 1
                original_rank = current_rankings[cat].index(team.team_id) + 1
                ranking_impact[cat] = self.league_size - original_rank
            else:
                rank = current_rankings[cat].index(team.team_id) + 1
                punt_points += (self.league_size + 1 - rank)
                ranking_impact[cat] = 0

        points_sacrificed = current_points - punt_points

        # Evaluate viability
        # Good punt strategies sacrifice few points in already-weak categories
        weak_punt = all(
            current_rankings[cat].index(team.team_id) + 1 >= self.league_size * 0.7
            for cat in punt_categories
        )

        if weak_punt and points_sacrificed <= len(punt_categories) * 2:
            viability = 80
            recommendation = f"Viable punt strategy. You're already weak in {', '.join(punt_categories)}. " \
                           f"Focus resources on improving other categories."
        elif points_sacrificed <= len(punt_categories) * 3:
            viability = 60
            recommendation = f"Moderate punt strategy. You would sacrifice {points_sacrificed:.1f} points. " \
                           f"Consider if the roster flexibility is worth it."
        else:
            viability = 30
            recommendation = f"Risky punt strategy. You would sacrifice {points_sacrificed:.1f} points " \
                           f"from categories where you're currently competitive."

        resources_freed = self._describe_freed_resources(punt_categories)

        return PuntStrategy(
            punt_categories=punt_categories,
            points_sacrificed=points_sacrificed,
            resources_freed=resources_freed,
            current_points=current_points,
            projected_points_with_punt=punt_points,
            ranking_impact=ranking_impact,
            viability_score=viability,
            recommendation=recommendation
        )

    # =========================================================================
    # Ranking Calculations
    # =========================================================================

    def _calculate_current_rankings(
        self,
        teams: List[TeamRotoStats]
    ) -> Dict[str, List[int]]:
        """
        Calculate current rankings for each category.

        Returns dict of category -> list of team_ids (sorted by rank, 1st to last)
        """
        rankings = {}

        for cat in self.categories:
            # Get values for all teams
            team_values = [
                (t.team_id, t.current_totals.get(cat, 0))
                for t in teams
            ]

            # Sort by value (descending for positive cats, ascending for negative)
            reverse = cat not in self.negative_categories
            team_values.sort(key=lambda x: x[1], reverse=reverse)

            rankings[cat] = [tv[0] for tv in team_values]

        return rankings

    def _calculate_projected_rankings(
        self,
        projected_totals: Dict[int, Dict[str, float]]
    ) -> Dict[str, List[int]]:
        """Calculate projected rankings based on projected totals."""
        rankings = {}

        for cat in self.categories:
            team_values = [
                (team_id, totals.get(cat, 0))
                for team_id, totals in projected_totals.items()
            ]

            reverse = cat not in self.negative_categories
            team_values.sort(key=lambda x: x[1], reverse=reverse)

            rankings[cat] = [tv[0] for tv in team_values]

        return rankings

    def _calculate_roto_points(
        self,
        team_id: int,
        rankings: Dict[str, List[int]]
    ) -> float:
        """Calculate total Roto points for a team."""
        total = 0
        for cat in self.categories:
            rank = rankings[cat].index(team_id) + 1
            points = self.league_size + 1 - rank
            total += points
        return total

    def _get_overall_rank(
        self,
        team_id: int,
        teams: List[TeamRotoStats],
        rankings: Dict[str, List[int]]
    ) -> int:
        """Get overall league rank for a team."""
        team_points = []
        for team in teams:
            points = self._calculate_roto_points(team.team_id, rankings)
            team_points.append((team.team_id, points))

        team_points.sort(key=lambda x: x[1], reverse=True)

        for rank, (tid, _) in enumerate(team_points, 1):
            if tid == team_id:
                return rank

        return len(teams)

    # =========================================================================
    # Simulation
    # =========================================================================

    def _simulate_season(
        self,
        teams: List[TeamRotoStats],
        projected_totals: Dict[int, Dict[str, float]]
    ) -> Dict[str, Dict[int, List]]:
        """
        Run Monte Carlo simulation for end-of-season standings.

        Returns dict with 'points', 'ranks', 'wins' for each team.
        """
        results = {
            'points': {t.team_id: [] for t in teams},
            'ranks': {t.team_id: [] for t in teams},
            'wins': {t.team_id: 0 for t in teams}
        }

        for _ in range(self.num_simulations):
            # Sample final totals with variance
            sampled_totals = {}
            for team in teams:
                sampled = {}
                base_totals = projected_totals[team.team_id]

                for cat in self.categories:
                    base_value = base_totals.get(cat, 0)
                    variance = PROJECTION_VARIANCE.get(cat, 0.10)
                    std = base_value * variance

                    value = self.rng.normal(base_value, std)

                    # Ensure valid range
                    if cat in RATE_CATEGORIES:
                        value = np.clip(value, 0, 1)
                    else:
                        value = max(0, value)

                    sampled[cat] = value

                sampled_totals[team.team_id] = sampled

            # Calculate rankings for this simulation
            sim_rankings = self._calculate_projected_rankings(sampled_totals)

            # Calculate points and ranks
            team_points = []
            for team in teams:
                points = self._calculate_roto_points(team.team_id, sim_rankings)
                team_points.append((team.team_id, points))
                results['points'][team.team_id].append(points)

            # Sort to determine ranks
            team_points.sort(key=lambda x: x[1], reverse=True)
            for rank, (tid, _) in enumerate(team_points, 1):
                results['ranks'][tid].append(rank)
                if rank == 1:
                    results['wins'][tid] += 1

        return results

    def _calculate_rank_distribution(
        self,
        ranks: List[int]
    ) -> Dict[int, float]:
        """Calculate probability distribution of final ranks."""
        distribution = defaultdict(int)
        for r in ranks:
            distribution[r] += 1

        total = len(ranks)
        return {r: count / total for r, count in sorted(distribution.items())}

    # =========================================================================
    # Category Analysis
    # =========================================================================

    def _analyze_categories(
        self,
        team: TeamRotoStats,
        all_teams: List[TeamRotoStats],
        current_rankings: Dict[str, List[int]],
        projected_rankings: Dict[str, List[int]],
        projected_totals: Dict[int, Dict[str, float]]
    ) -> Dict[str, CategoryRanking]:
        """Analyze each category for a team."""
        team_lookup = {t.team_id: t for t in all_teams}
        category_rankings = {}

        for cat in self.categories:
            current_rank = current_rankings[cat].index(team.team_id) + 1
            projected_rank = projected_rankings[cat].index(team.team_id) + 1

            current_value = team.current_totals.get(cat, 0)
            projected_value = projected_totals[team.team_id].get(cat, 0)

            # Gap analysis
            my_idx = projected_rankings[cat].index(team.team_id)

            if my_idx > 0:
                ahead_id = projected_rankings[cat][my_idx - 1]
                ahead_value = projected_totals[ahead_id].get(cat, 0)
                next_team = team_lookup[ahead_id].team_name

                if cat in self.negative_categories:
                    gap_to_next = projected_value - ahead_value
                else:
                    gap_to_next = ahead_value - projected_value
            else:
                gap_to_next = 0
                next_team = None

            if my_idx < len(projected_rankings[cat]) - 1:
                behind_id = projected_rankings[cat][my_idx + 1]
                behind_value = projected_totals[behind_id].get(cat, 0)
                prev_team = team_lookup[behind_id].team_name

                if cat in self.negative_categories:
                    gap_to_prev = behind_value - projected_value
                else:
                    gap_to_prev = projected_value - behind_value
            else:
                gap_to_prev = float('inf')
                prev_team = None

            # Determine strength
            if projected_rank <= 2:
                strength = CategoryStrength.DOMINANT
            elif projected_rank <= 4:
                strength = CategoryStrength.STRONG
            elif projected_rank <= self.league_size - 4:
                strength = CategoryStrength.AVERAGE
            elif projected_rank <= self.league_size - 2:
                strength = CategoryStrength.WEAK
            else:
                strength = CategoryStrength.PUNT

            # Rank probability (simplified - would need simulation for accuracy)
            rank_prob = {projected_rank: 0.6}
            if projected_rank > 1:
                rank_prob[projected_rank - 1] = 0.2
            if projected_rank < self.league_size:
                rank_prob[projected_rank + 1] = 0.2

            category_rankings[cat] = CategoryRanking(
                category=cat,
                current_rank=current_rank,
                projected_rank=projected_rank,
                rank_probability=rank_prob,
                current_value=current_value,
                projected_value=projected_value,
                gap_to_next=gap_to_next,
                gap_to_previous=gap_to_prev,
                next_team=next_team,
                previous_team=prev_team,
                strength=strength
            )

        return category_rankings

    def _identify_punt_candidates(
        self,
        category_rankings: Dict[str, CategoryRanking],
        current_points: float
    ) -> List[str]:
        """Identify categories that are good punt candidates."""
        candidates = []

        for cat, ranking in category_rankings.items():
            # Good punt candidate if:
            # 1. Already weak (rank >= 10 in 12-team league)
            # 2. Gap to improve is large
            # 3. Not a high-value category

            if ranking.strength in [CategoryStrength.WEAK, CategoryStrength.PUNT]:
                if ranking.gap_to_next > ranking.projected_value * 0.1:  # >10% gap
                    candidates.append(cat)

        return candidates

    def _identify_improvement_targets(
        self,
        team: TeamRotoStats,
        all_teams: List[TeamRotoStats],
        category_rankings: Dict[str, CategoryRanking],
        projected_totals: Dict[int, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Identify categories where small improvements yield ranking gains."""
        targets = []

        for cat, ranking in category_rankings.items():
            # Skip if already dominant
            if ranking.strength == CategoryStrength.DOMINANT:
                continue

            # Calculate improvement needed
            if ranking.gap_to_next <= 0:
                continue  # Already improving

            # Relative difficulty
            relative_gap = ranking.gap_to_next / max(ranking.projected_value, 1)

            # Games of production needed
            ros_rate = team.ros_projections.get(cat, 1)
            if ros_rate > 0:
                games_needed = ranking.gap_to_next / (ros_rate * team.roster_size)
            else:
                games_needed = float('inf')

            if games_needed <= team.games_remaining * 0.5:  # Achievable
                targets.append({
                    'category': cat,
                    'current_rank': ranking.current_rank,
                    'gap_to_improve': round(ranking.gap_to_next, 2),
                    'games_needed': int(games_needed),
                    'difficulty': 'easy' if games_needed < team.games_remaining * 0.2 else 'moderate',
                    'next_team': ranking.next_team,
                    'points_gain': 1  # Moving up one rank
                })

        # Sort by difficulty (easiest first)
        targets.sort(key=lambda x: x['games_needed'])

        return targets[:5]  # Top 5 targets

    def _describe_freed_resources(self, punt_categories: List[str]) -> str:
        """Describe what resources are freed by punting categories."""
        descriptions = []

        if 'ast' in punt_categories:
            descriptions.append("traditional point guards")
        if 'blk' in punt_categories:
            descriptions.append("shot-blocking centers")
        if 'stl' in punt_categories:
            descriptions.append("perimeter defenders")
        if '3p' in punt_categories:
            descriptions.append("high-volume shooters")
        if 'pts' in punt_categories:
            descriptions.append("high-usage scorers")
        if 'trb' in punt_categories:
            descriptions.append("rebounding specialists")
        if 'ft_pct' in punt_categories:
            descriptions.append("can target poor FT shooters (often big men)")
        if 'fg_pct' in punt_categories:
            descriptions.append("can target high-volume shooters")
        if 'tov' in punt_categories:
            descriptions.append("high-usage ball handlers")

        if descriptions:
            return f"Can deprioritize {', '.join(descriptions)}"
        return "Minimal roster impact"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_roto_analyzer(
    league_size: int = DEFAULT_LEAGUE_SIZE,
    categories: Optional[List[str]] = None,
    num_simulations: int = DEFAULT_SIMULATIONS
) -> RotoAnalyzer:
    """
    Factory function to create a configured Roto analyzer.

    Args:
        league_size: Number of teams
        categories: Custom categories
        num_simulations: Monte Carlo iterations

    Returns:
        Configured RotoAnalyzer
    """
    return RotoAnalyzer(
        league_size=league_size,
        categories=categories,
        num_simulations=num_simulations
    )


def quick_roto_analysis(
    team_totals: Dict[str, float],
    league_totals: List[Dict[str, float]],
    team_name: str = "Your Team"
) -> Dict[str, Any]:
    """
    Quick Roto standings analysis.

    Args:
        team_totals: Your team's current totals
        league_totals: List of all teams' totals
        team_name: Your team name

    Returns:
        Analysis summary
    """
    # Build team objects
    teams = []
    for i, totals in enumerate(league_totals):
        teams.append(TeamRotoStats(
            team_id=i,
            team_name=f"Team {i+1}" if totals != team_totals else team_name,
            current_totals=totals,
            ros_projections={k: v / 50 for k, v in totals.items()}  # Rough per-game
        ))

    analyzer = RotoAnalyzer(league_size=len(teams), num_simulations=5000)

    # Find user's team
    user_team = next(t for t in teams if t.team_name == team_name)
    projection = analyzer.analyze_team(user_team, teams)

    return {
        'team_name': team_name,
        'current_points': projection.current_points,
        'current_rank': projection.current_rank,
        'projected_points': projection.projected_points,
        'projected_rank': projection.projected_rank,
        'win_probability': f"{projection.win_probability:.1%}",
        'strengths': projection.strengths,
        'weaknesses': projection.weaknesses,
        'punt_candidates': projection.punt_candidates,
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Demo/test entry point for Roto analysis."""
    logger.info("=" * 60)
    logger.info("Rotisserie League Analyzer Demo")
    logger.info("=" * 60)

    # Create analyzer for 10-team league
    analyzer = RotoAnalyzer(league_size=10, num_simulations=10000)

    # Example teams (10-team league, mid-season totals)
    teams = [
        TeamRotoStats(
            team_id=1, team_name="My Team",
            current_totals={'pts': 4200, 'trb': 1800, 'ast': 1100, 'stl': 320, 'blk': 220,
                          '3p': 480, 'fg_pct': 0.472, 'ft_pct': 0.815, 'tov': 520},
            fgm=1580, fga=3347, ftm=680, fta=834,
            ros_projections={'pts': 48.0, 'trb': 20.0, 'ast': 12.5, 'stl': 3.6, 'blk': 2.5,
                           '3p': 5.5, 'fgm': 18.0, 'fga': 38.0, 'ftm': 7.5, 'fta': 9.2, 'tov': 5.8},
            games_remaining=35
        ),
        TeamRotoStats(
            team_id=2, team_name="Team Alpha",
            current_totals={'pts': 4500, 'trb': 1650, 'ast': 1250, 'stl': 340, 'blk': 180,
                          '3p': 520, 'fg_pct': 0.465, 'ft_pct': 0.795, 'tov': 580},
            fgm=1650, fga=3548, ftm=720, fta=905,
            ros_projections={'pts': 52.0, 'trb': 18.0, 'ast': 14.0, 'stl': 3.8, 'blk': 2.0,
                           '3p': 6.0, 'fgm': 19.0, 'fga': 41.0, 'ftm': 8.0, 'fta': 10.0, 'tov': 6.5},
            games_remaining=35
        ),
        TeamRotoStats(
            team_id=3, team_name="Team Beta",
            current_totals={'pts': 3900, 'trb': 2100, 'ast': 900, 'stl': 280, 'blk': 320,
                          '3p': 380, 'fg_pct': 0.488, 'ft_pct': 0.720, 'tov': 450},
            fgm=1520, fga=3115, ftm=580, fta=805,
            ros_projections={'pts': 44.0, 'trb': 24.0, 'ast': 10.0, 'stl': 3.2, 'blk': 3.6,
                           '3p': 4.2, 'fgm': 17.0, 'fga': 35.0, 'ftm': 6.5, 'fta': 9.0, 'tov': 5.0},
            games_remaining=35
        ),
    ]

    # Add more teams for a fuller league
    for i in range(4, 11):
        teams.append(TeamRotoStats(
            team_id=i, team_name=f"Team {i}",
            current_totals={
                'pts': 3800 + np.random.randint(-400, 600),
                'trb': 1700 + np.random.randint(-300, 400),
                'ast': 1000 + np.random.randint(-200, 300),
                'stl': 300 + np.random.randint(-50, 60),
                'blk': 200 + np.random.randint(-60, 100),
                '3p': 420 + np.random.randint(-80, 120),
                'fg_pct': 0.460 + np.random.uniform(-0.02, 0.03),
                'ft_pct': 0.780 + np.random.uniform(-0.05, 0.05),
                'tov': 500 + np.random.randint(-80, 100)
            },
            fgm=1500, fga=3200, ftm=650, fta=820,
            ros_projections={'pts': 45.0, 'trb': 19.0, 'ast': 11.0, 'stl': 3.4, 'blk': 2.3,
                           '3p': 4.8, 'fgm': 17.0, 'fga': 37.0, 'ftm': 7.0, 'fta': 9.0, 'tov': 5.5},
            games_remaining=35
        ))

    # Analyze my team
    my_team = teams[0]
    projection = analyzer.analyze_team(my_team, teams)

    print("\n" + "=" * 60)
    print(f"Roto Analysis: {projection.team_name}")
    print("=" * 60)

    print(f"\nCurrent Standing: {projection.current_rank} of {analyzer.league_size}")
    print(f"Current Points: {projection.current_points:.1f}")
    print(f"\nProjected Standing: {projection.projected_rank:.1f}")
    print(f"Projected Points: {projection.projected_points:.1f}")
    print(f"\nWin Probability: {projection.win_probability:.1%}")
    print(f"Top 3 Probability: {projection.top_3_probability:.1%}")

    print("\nCategory Rankings:")
    print("-" * 70)
    print(f"{'Category':<10} {'Current':<10} {'Projected':<10} {'Strength':<12} {'Gap to Next':<15}")
    print("-" * 70)

    for cat in STANDARD_CATEGORIES:
        ranking = projection.category_rankings[cat]
        gap_str = f"{ranking.gap_to_next:+.1f}" if ranking.gap_to_next != 0 else "-"
        print(f"{cat.upper():<10} {ranking.current_rank:<10} {ranking.projected_rank:<10.1f} "
              f"{ranking.strength.value:<12} {gap_str:<15}")

    print("\nStrengths:", ", ".join(s.upper() for s in projection.strengths) or "None")
    print("Weaknesses:", ", ".join(w.upper() for w in projection.weaknesses) or "None")

    if projection.punt_candidates:
        print("\nPunt Candidates:", ", ".join(c.upper() for c in projection.punt_candidates))

    if projection.improvement_targets:
        print("\nImprovement Targets:")
        for target in projection.improvement_targets[:3]:
            print(f"  - {target['category'].upper()}: Gap of {target['gap_to_improve']:.1f} "
                  f"({target['games_needed']} games needed) - {target['difficulty']}")

    # Gap analysis
    print("\n" + "=" * 60)
    print("Gap Analysis")
    print("=" * 60)

    gaps = analyzer.get_gap_analysis(my_team, teams)
    for cat in ['pts', 'trb', 'ast']:
        gap = gaps[cat]
        print(f"\n{cat.upper()}:")
        print(f"  Current Rank: {gap.current_rank}")
        print(f"  Gap to improve: {gap.gap_to_improve:+.1f} (vs {gap.team_ahead})")
        print(f"  Cushion: {gap.cushion:.1f} (vs {gap.team_behind})")
        print(f"  Difficulty: {gap.improvement_difficulty}")

    # Evaluate punt strategy
    print("\n" + "=" * 60)
    print("Punt Strategy Evaluation")
    print("=" * 60)

    if projection.punt_candidates:
        punt_eval = analyzer.evaluate_punt_strategy(my_team, teams, projection.punt_candidates[:2])
        print(f"\nPunting: {', '.join(c.upper() for c in punt_eval.punt_categories)}")
        print(f"Points Sacrificed: {punt_eval.points_sacrificed:.1f}")
        print(f"Viability Score: {punt_eval.viability_score:.0f}/100")
        print(f"Recommendation: {punt_eval.recommendation}")
        print(f"Resources Freed: {punt_eval.resources_freed}")

    print("\n" + "=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())

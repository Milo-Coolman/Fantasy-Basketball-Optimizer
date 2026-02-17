#!/usr/bin/env python3
"""
H2H Matchup Analyzer for Fantasy Basketball.

This module provides Monte Carlo simulation-based analysis for Head-to-Head
Category fantasy basketball leagues, including:
- Season simulation with playoff probability
- Category-by-category win probability
- Weekly matchup projections
- Strength of schedule analysis
- Streaming recommendations

Reference: PRD Section 3.3.5 - H2H League Projections
"""

import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import random

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

# Default simulation settings
DEFAULT_SIMULATIONS = 10000
MIN_SIMULATIONS = 1000
MAX_SIMULATIONS = 100000

# Standard H2H categories (9-cat)
STANDARD_CATEGORIES = [
    'pts', 'trb', 'ast', 'stl', 'blk', '3p', 'fg_pct', 'ft_pct', 'tov'
]

# Categories where lower is better
NEGATIVE_CATEGORIES = ['tov']

# Default playoff settings
DEFAULT_PLAYOFF_TEAMS = 6
DEFAULT_PLAYOFF_WEEKS = 3
DEFAULT_REGULAR_SEASON_WEEKS = 18

# Variance scaling for stat projections (used in Monte Carlo)
STAT_VARIANCE_FACTORS = {
    'pts': 0.12,     # 12% variance
    'trb': 0.15,
    'ast': 0.18,
    'stl': 0.25,     # Higher variance for low-volume stats
    'blk': 0.28,
    'tov': 0.20,
    '3p': 0.22,
    'fg_pct': 0.04,  # Lower variance for percentages
    'ft_pct': 0.05,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TeamProjection:
    """Projected stats for a team in a given week."""
    team_id: int
    team_name: str
    projected_stats: Dict[str, float]
    stat_variance: Dict[str, float] = field(default_factory=dict)
    games_in_week: int = 3  # Average games per player per week
    roster_size: int = 10

    def sample_stats(self, rng: np.random.Generator) -> Dict[str, float]:
        """
        Sample stats using normal distribution with variance.

        Args:
            rng: NumPy random generator for reproducibility

        Returns:
            Sampled stats dictionary
        """
        sampled = {}
        for stat, mean in self.projected_stats.items():
            variance = self.stat_variance.get(stat, STAT_VARIANCE_FACTORS.get(stat, 0.15))
            std = mean * variance

            # Sample from normal distribution
            value = rng.normal(mean, std)

            # Ensure non-negative for counting stats
            if stat not in ['fg_pct', 'ft_pct']:
                value = max(0, value)
            else:
                # Percentages bounded 0-1
                value = np.clip(value, 0, 1)

            sampled[stat] = value

        return sampled


@dataclass
class MatchupResult:
    """Result of a single matchup between two teams."""
    team1_id: int
    team2_id: int
    team1_wins: int
    team2_wins: int
    ties: int
    category_results: Dict[str, str]  # 'team1', 'team2', or 'tie'

    @property
    def winner(self) -> Optional[int]:
        """Return winning team ID or None for tie."""
        if self.team1_wins > self.team2_wins:
            return self.team1_id
        elif self.team2_wins > self.team1_wins:
            return self.team2_id
        return None

    @property
    def result_string(self) -> str:
        """Return formatted result string."""
        return f"{self.team1_wins}-{self.team2_wins}-{self.ties}"


@dataclass
class WeeklyMatchupProjection:
    """Projection for a weekly H2H matchup."""
    week: int
    team_id: int
    opponent_id: int
    team_name: str
    opponent_name: str

    # Category projections
    category_win_probs: Dict[str, float]
    expected_category_wins: float
    expected_category_losses: float
    expected_ties: float

    # Overall matchup
    matchup_win_prob: float
    projected_result: str  # e.g., "5-3-1"

    # Recommended actions
    streaming_targets: List[Dict[str, Any]] = field(default_factory=list)
    punt_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'week': self.week,
            'team_id': self.team_id,
            'opponent_id': self.opponent_id,
            'team_name': self.team_name,
            'opponent_name': self.opponent_name,
            'category_win_probs': {k: round(v, 3) for k, v in self.category_win_probs.items()},
            'expected_category_wins': round(self.expected_category_wins, 2),
            'matchup_win_prob': round(self.matchup_win_prob, 3),
            'projected_result': self.projected_result,
            'streaming_targets': self.streaming_targets,
            'punt_categories': self.punt_categories,
        }


@dataclass
class SeasonSimulationResult:
    """Results from Monte Carlo season simulation."""
    team_id: int
    team_name: str

    # Record projections
    current_wins: int
    current_losses: int
    current_ties: int

    projected_wins: float
    projected_losses: float
    projected_ties: float

    # Standings
    current_standing: int
    projected_standing: float
    standing_distribution: Dict[int, float]  # standing -> probability

    # Playoff
    playoff_probability: float
    bye_probability: float  # Top 2 often get bye
    championship_probability: float

    # Category strengths
    category_rankings: Dict[str, float]  # category -> avg league rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            'team_id': self.team_id,
            'team_name': self.team_name,
            'current_record': f"{self.current_wins}-{self.current_losses}-{self.current_ties}",
            'projected_record': f"{self.projected_wins:.1f}-{self.projected_losses:.1f}-{self.projected_ties:.1f}",
            'current_standing': self.current_standing,
            'projected_standing': round(self.projected_standing, 1),
            'playoff_probability': round(self.playoff_probability * 100, 1),
            'bye_probability': round(self.bye_probability * 100, 1),
            'championship_probability': round(self.championship_probability * 100, 1),
            'category_rankings': {k: round(v, 1) for k, v in self.category_rankings.items()},
        }


@dataclass
class ScheduleStrength:
    """Strength of schedule metrics for remaining matchups."""
    team_id: int
    team_name: str
    remaining_weeks: int
    remaining_opponents: List[int]

    # Strength metrics
    avg_opponent_rank: float
    avg_opponent_win_pct: float
    strength_rating: float  # 0-100, higher = harder

    # Easy/hard weeks
    easiest_week: int
    hardest_week: int

    # Opponent breakdown
    opponent_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamingRecommendation:
    """Recommendation for a streaming pickup."""
    player_id: str
    player_name: str
    team: str
    position: str

    # Schedule info
    games_this_week: int
    games_remaining_week: int

    # Impact
    category_impacts: Dict[str, float]
    priority_categories: List[str]
    overall_impact_score: float

    # Acquisition info
    is_available: bool
    ownership_pct: float


# =============================================================================
# Matchup Analyzer
# =============================================================================

class MatchupAnalyzer:
    """
    Analyzer for H2H Category fantasy basketball matchups.

    Uses Monte Carlo simulation to project matchup outcomes,
    playoff probabilities, and provide strategic recommendations.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        negative_categories: Optional[List[str]] = None,
        num_simulations: int = DEFAULT_SIMULATIONS,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the matchup analyzer.

        Args:
            categories: Stat categories for H2H (default: 9-cat)
            negative_categories: Categories where lower is better
            num_simulations: Number of Monte Carlo iterations
            random_seed: Seed for reproducibility
        """
        self.categories = categories or STANDARD_CATEGORIES.copy()
        self.negative_categories = set(negative_categories or NEGATIVE_CATEGORIES)
        self.num_simulations = np.clip(num_simulations, MIN_SIMULATIONS, MAX_SIMULATIONS)

        # Initialize random generator
        self.rng = np.random.default_rng(random_seed)

        # Caches
        self._team_projections: Dict[int, TeamProjection] = {}
        self._schedule_cache: Dict[int, List[Tuple[int, int]]] = {}  # team_id -> [(week, opponent_id)]

    # =========================================================================
    # Single Matchup Analysis
    # =========================================================================

    def analyze_matchup(
        self,
        team1: TeamProjection,
        team2: TeamProjection,
        num_simulations: Optional[int] = None
    ) -> Tuple[Dict[str, float], float, MatchupResult]:
        """
        Analyze a single matchup between two teams.

        Args:
            team1: First team's projection
            team2: Second team's projection
            num_simulations: Override default simulation count

        Returns:
            Tuple of (category_win_probs, overall_win_prob, expected_result)
        """
        n_sims = num_simulations or self.num_simulations

        # Track wins in each category
        category_wins = {cat: 0 for cat in self.categories}
        matchup_wins = 0
        matchup_ties = 0

        for _ in range(n_sims):
            # Sample stats for both teams
            team1_stats = team1.sample_stats(self.rng)
            team2_stats = team2.sample_stats(self.rng)

            # Count category wins
            t1_cats = 0
            t2_cats = 0
            ties = 0

            for cat in self.categories:
                t1_val = team1_stats.get(cat, 0)
                t2_val = team2_stats.get(cat, 0)

                # Determine winner (flip for negative categories)
                if cat in self.negative_categories:
                    t1_val, t2_val = -t1_val, -t2_val

                if t1_val > t2_val:
                    category_wins[cat] += 1
                    t1_cats += 1
                elif t2_val > t1_val:
                    t2_cats += 1
                else:
                    ties += 1

            # Determine matchup winner
            if t1_cats > t2_cats:
                matchup_wins += 1
            elif t1_cats == t2_cats:
                matchup_ties += 1

        # Calculate probabilities
        category_win_probs = {
            cat: wins / n_sims for cat, wins in category_wins.items()
        }
        overall_win_prob = matchup_wins / n_sims

        # Calculate expected result
        exp_wins = sum(category_win_probs.values())
        exp_losses = len(self.categories) - exp_wins - sum(
            1 for p in category_win_probs.values() if 0.45 <= p <= 0.55
        ) * 0.5
        exp_ties = len(self.categories) - exp_wins - exp_losses

        # Build expected matchup result
        cat_results = {}
        for cat, prob in category_win_probs.items():
            if prob > 0.55:
                cat_results[cat] = 'team1'
            elif prob < 0.45:
                cat_results[cat] = 'team2'
            else:
                cat_results[cat] = 'tie'

        expected_result = MatchupResult(
            team1_id=team1.team_id,
            team2_id=team2.team_id,
            team1_wins=int(round(exp_wins)),
            team2_wins=int(round(len(self.categories) - exp_wins - max(0, exp_ties))),
            ties=int(round(max(0, exp_ties))),
            category_results=cat_results
        )

        return category_win_probs, overall_win_prob, expected_result

    def project_weekly_matchup(
        self,
        team: TeamProjection,
        opponent: TeamProjection,
        week: int,
        available_streamers: Optional[List[Dict[str, Any]]] = None
    ) -> WeeklyMatchupProjection:
        """
        Generate detailed projection for a weekly matchup.

        Args:
            team: User's team projection
            opponent: Opponent's team projection
            week: Week number
            available_streamers: List of available streaming options

        Returns:
            WeeklyMatchupProjection with win probabilities and recommendations
        """
        # Analyze the matchup
        cat_probs, win_prob, result = self.analyze_matchup(team, opponent)

        # Calculate expected categories
        exp_wins = sum(cat_probs.values())
        exp_losses = sum(1 - p for p in cat_probs.values())
        exp_ties = len(self.categories) - exp_wins - exp_losses

        # Identify punt categories (< 30% win probability)
        punt_cats = [cat for cat, prob in cat_probs.items() if prob < 0.30]

        # Generate streaming recommendations
        streaming_targets = []
        if available_streamers:
            streaming_targets = self._rank_streaming_options(
                team, opponent, cat_probs, available_streamers
            )

        return WeeklyMatchupProjection(
            week=week,
            team_id=team.team_id,
            opponent_id=opponent.team_id,
            team_name=team.team_name,
            opponent_name=opponent.team_name,
            category_win_probs=cat_probs,
            expected_category_wins=exp_wins,
            expected_category_losses=exp_losses,
            expected_ties=max(0, exp_ties),
            matchup_win_prob=win_prob,
            projected_result=result.result_string,
            streaming_targets=streaming_targets[:5],  # Top 5
            punt_categories=punt_cats
        )

    # =========================================================================
    # Season Simulation
    # =========================================================================

    def simulate_season(
        self,
        teams: List[TeamProjection],
        schedule: Dict[int, List[Tuple[int, int]]],  # team_id -> [(week, opponent_id)]
        current_standings: Dict[int, Tuple[int, int, int]],  # team_id -> (wins, losses, ties)
        current_week: int,
        total_weeks: int = DEFAULT_REGULAR_SEASON_WEEKS,
        playoff_teams: int = DEFAULT_PLAYOFF_TEAMS,
        num_simulations: Optional[int] = None
    ) -> List[SeasonSimulationResult]:
        """
        Run Monte Carlo simulation for the rest of the season.

        Args:
            teams: List of team projections
            schedule: Remaining schedule for each team
            current_standings: Current win/loss/tie for each team
            current_week: Current week of the season
            total_weeks: Total weeks in regular season
            playoff_teams: Number of teams making playoffs
            num_simulations: Override default simulation count

        Returns:
            List of SeasonSimulationResult for each team
        """
        n_sims = num_simulations or self.num_simulations

        # Create team lookup
        team_lookup = {t.team_id: t for t in teams}
        team_ids = [t.team_id for t in teams]

        # Initialize tracking
        final_standings = {tid: [] for tid in team_ids}  # List of final standings per sim
        final_records = {tid: {'w': [], 'l': [], 't': []} for tid in team_ids}
        playoff_made = {tid: 0 for tid in team_ids}
        bye_earned = {tid: 0 for tid in team_ids}
        championship_won = {tid: 0 for tid in team_ids}

        logger.info(f"Running {n_sims} season simulations from week {current_week}...")

        for sim in range(n_sims):
            # Copy current standings
            sim_records = {
                tid: list(current_standings.get(tid, (0, 0, 0)))
                for tid in team_ids
            }

            # Simulate remaining weeks
            for week in range(current_week, total_weeks + 1):
                # Process each matchup this week
                matchups_this_week = set()

                for team_id in team_ids:
                    team_schedule = schedule.get(team_id, [])

                    # Find opponent for this week
                    for sched_week, opp_id in team_schedule:
                        if sched_week == week and (opp_id, team_id) not in matchups_this_week:
                            matchups_this_week.add((team_id, opp_id))

                # Simulate each matchup
                for t1_id, t2_id in matchups_this_week:
                    t1 = team_lookup.get(t1_id)
                    t2 = team_lookup.get(t2_id)

                    if not t1 or not t2:
                        continue

                    # Simulate single matchup
                    t1_stats = t1.sample_stats(self.rng)
                    t2_stats = t2.sample_stats(self.rng)

                    t1_cats, t2_cats = 0, 0
                    for cat in self.categories:
                        v1 = t1_stats.get(cat, 0)
                        v2 = t2_stats.get(cat, 0)

                        if cat in self.negative_categories:
                            v1, v2 = -v1, -v2

                        if v1 > v2:
                            t1_cats += 1
                        elif v2 > v1:
                            t2_cats += 1

                    # Update records
                    if t1_cats > t2_cats:
                        sim_records[t1_id][0] += 1
                        sim_records[t2_id][1] += 1
                    elif t2_cats > t1_cats:
                        sim_records[t2_id][0] += 1
                        sim_records[t1_id][1] += 1
                    else:
                        sim_records[t1_id][2] += 1
                        sim_records[t2_id][2] += 1

            # Determine final standings
            standings = sorted(
                team_ids,
                key=lambda tid: (
                    sim_records[tid][0],  # Wins
                    -sim_records[tid][1],  # Losses (negative)
                    sim_records[tid][2]   # Ties
                ),
                reverse=True
            )

            # Record results for this simulation
            for rank, tid in enumerate(standings, 1):
                final_standings[tid].append(rank)
                final_records[tid]['w'].append(sim_records[tid][0])
                final_records[tid]['l'].append(sim_records[tid][1])
                final_records[tid]['t'].append(sim_records[tid][2])

                # Playoff tracking
                if rank <= playoff_teams:
                    playoff_made[tid] += 1
                if rank <= 2:
                    bye_earned[tid] += 1

            # Simulate playoffs (simplified)
            playoff_bracket = standings[:playoff_teams]
            if len(playoff_bracket) >= 2:
                # Championship is between top seeds after bracket
                # Simplified: weight by regular season performance
                champ_weights = [1.0 / (i + 1) for i in range(len(playoff_bracket))]
                total = sum(champ_weights)
                champ_probs = [w / total for w in champ_weights]
                champ_idx = self.rng.choice(len(playoff_bracket), p=champ_probs)
                championship_won[playoff_bracket[champ_idx]] += 1

        # Build results
        results = []
        for team in teams:
            tid = team.team_id
            curr = current_standings.get(tid, (0, 0, 0))

            # Calculate standing distribution
            standing_counts = defaultdict(int)
            for s in final_standings[tid]:
                standing_counts[s] += 1
            standing_dist = {s: count / n_sims for s, count in standing_counts.items()}

            # Calculate category rankings (simplified - based on projections)
            cat_rankings = self._calculate_category_rankings(team, teams)

            results.append(SeasonSimulationResult(
                team_id=tid,
                team_name=team.team_name,
                current_wins=curr[0],
                current_losses=curr[1],
                current_ties=curr[2],
                projected_wins=np.mean(final_records[tid]['w']),
                projected_losses=np.mean(final_records[tid]['l']),
                projected_ties=np.mean(final_records[tid]['t']),
                current_standing=self._get_current_standing(tid, current_standings, team_ids),
                projected_standing=np.mean(final_standings[tid]),
                standing_distribution=standing_dist,
                playoff_probability=playoff_made[tid] / n_sims,
                bye_probability=bye_earned[tid] / n_sims,
                championship_probability=championship_won[tid] / n_sims,
                category_rankings=cat_rankings
            ))

        # Sort by projected standing
        results.sort(key=lambda r: r.projected_standing)

        return results

    # =========================================================================
    # Schedule Analysis
    # =========================================================================

    def analyze_schedule_strength(
        self,
        team_id: int,
        teams: List[TeamProjection],
        schedule: List[Tuple[int, int]],  # [(week, opponent_id)]
        current_standings: Dict[int, Tuple[int, int, int]]
    ) -> ScheduleStrength:
        """
        Analyze strength of remaining schedule.

        Args:
            team_id: Team to analyze
            teams: All team projections
            schedule: Team's remaining schedule
            current_standings: Current standings

        Returns:
            ScheduleStrength analysis
        """
        team_lookup = {t.team_id: t for t in teams}
        team = team_lookup.get(team_id)

        if not team:
            raise ValueError(f"Team {team_id} not found")

        # Calculate opponent strength metrics
        opponent_ranks = []
        opponent_win_pcts = []
        opponent_details = []

        total_teams = len(teams)

        for week, opp_id in schedule:
            opp = team_lookup.get(opp_id)
            if not opp:
                continue

            # Get opponent's current standing
            opp_record = current_standings.get(opp_id, (0, 0, 0))
            total_games = sum(opp_record)
            win_pct = opp_record[0] / max(total_games, 1)

            # Estimate rank based on record
            opp_rank = self._get_current_standing(opp_id, current_standings, list(team_lookup.keys()))

            opponent_ranks.append(opp_rank)
            opponent_win_pcts.append(win_pct)

            opponent_details.append({
                'week': week,
                'opponent_id': opp_id,
                'opponent_name': opp.team_name,
                'opponent_rank': opp_rank,
                'opponent_win_pct': round(win_pct, 3),
                'difficulty': 'hard' if opp_rank <= total_teams // 3 else
                             'easy' if opp_rank >= 2 * total_teams // 3 else 'medium'
            })

        if not opponent_ranks:
            return ScheduleStrength(
                team_id=team_id,
                team_name=team.team_name,
                remaining_weeks=0,
                remaining_opponents=[],
                avg_opponent_rank=0,
                avg_opponent_win_pct=0,
                strength_rating=50,
                easiest_week=0,
                hardest_week=0,
                opponent_details=[]
            )

        # Calculate strength rating (0-100, higher = harder)
        avg_rank = np.mean(opponent_ranks)
        avg_win_pct = np.mean(opponent_win_pcts)

        # Normalize rank (1 is hardest, n is easiest)
        rank_score = (total_teams - avg_rank) / (total_teams - 1) * 100
        win_pct_score = avg_win_pct * 100

        strength_rating = (rank_score + win_pct_score) / 2

        # Find easiest and hardest weeks
        easiest_idx = np.argmax(opponent_ranks)
        hardest_idx = np.argmin(opponent_ranks)

        return ScheduleStrength(
            team_id=team_id,
            team_name=team.team_name,
            remaining_weeks=len(schedule),
            remaining_opponents=[opp_id for _, opp_id in schedule],
            avg_opponent_rank=avg_rank,
            avg_opponent_win_pct=avg_win_pct,
            strength_rating=strength_rating,
            easiest_week=schedule[easiest_idx][0] if schedule else 0,
            hardest_week=schedule[hardest_idx][0] if schedule else 0,
            opponent_details=opponent_details
        )

    # =========================================================================
    # Streaming Recommendations
    # =========================================================================

    def get_streaming_recommendations(
        self,
        team: TeamProjection,
        opponent: TeamProjection,
        available_players: List[Dict[str, Any]],
        games_remaining_this_week: int,
        target_categories: Optional[List[str]] = None
    ) -> List[StreamingRecommendation]:
        """
        Get streaming recommendations for current matchup.

        Args:
            team: User's team projection
            opponent: Opponent's projection
            available_players: List of available free agents
            games_remaining_this_week: Games left in the week
            target_categories: Optional specific categories to target

        Returns:
            Sorted list of streaming recommendations
        """
        # Analyze current matchup to identify needs
        cat_probs, _, _ = self.analyze_matchup(team, opponent)

        # Identify categories to target
        # Focus on close categories (40-60% win prob) and slightly losing ones
        if target_categories:
            priority_cats = target_categories
        else:
            priority_cats = [
                cat for cat, prob in cat_probs.items()
                if 0.30 <= prob <= 0.60 and cat not in self.negative_categories
            ]
            # Also consider turnovers if we're close to winning it
            if 'tov' in cat_probs and cat_probs['tov'] >= 0.40:
                priority_cats.append('tov')

        recommendations = []

        for player in available_players:
            player_id = player.get('player_id', player.get('id', ''))
            player_name = player.get('name', 'Unknown')
            player_team = player.get('team', 'UNK')
            position = player.get('position', 'N/A')
            games_this_week = player.get('games_this_week', 0)
            games_remaining = player.get('games_remaining_week', games_this_week)
            ownership = player.get('ownership_pct', 0)
            is_available = player.get('is_available', True)

            # Skip if no games remaining
            if games_remaining <= 0:
                continue

            # Calculate category impacts
            player_stats = player.get('projected_stats', player.get('stats', {}))
            category_impacts = {}

            for cat in self.categories:
                if cat in player_stats:
                    # Impact = player's contribution * games remaining
                    impact = player_stats[cat] * games_remaining

                    # Weight by category importance (closer matchups = more important)
                    if cat in priority_cats:
                        impact *= 1.5

                    # Turnovers are negative
                    if cat in self.negative_categories:
                        impact = -impact

                    category_impacts[cat] = impact

            # Calculate overall impact score
            overall_impact = sum(
                impact for cat, impact in category_impacts.items()
                if cat in priority_cats or impact > 0
            )

            # Bonus for more games
            overall_impact *= (1 + games_remaining * 0.1)

            recommendations.append(StreamingRecommendation(
                player_id=player_id,
                player_name=player_name,
                team=player_team,
                position=position,
                games_this_week=games_this_week,
                games_remaining_week=games_remaining,
                category_impacts=category_impacts,
                priority_categories=[c for c in priority_cats if c in category_impacts],
                overall_impact_score=overall_impact,
                is_available=is_available,
                ownership_pct=ownership
            ))

        # Sort by impact score
        recommendations.sort(key=lambda r: r.overall_impact_score, reverse=True)

        return recommendations

    def _rank_streaming_options(
        self,
        team: TeamProjection,
        opponent: TeamProjection,
        category_probs: Dict[str, float],
        available_players: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank streaming options for weekly matchup.

        Internal helper that returns simplified recommendation dicts.
        """
        # Get full recommendations
        recs = self.get_streaming_recommendations(
            team, opponent, available_players,
            games_remaining_this_week=3  # Assume mid-week
        )

        # Convert to simple dicts
        return [
            {
                'player_id': r.player_id,
                'player_name': r.player_name,
                'team': r.team,
                'position': r.position,
                'games_remaining': r.games_remaining_week,
                'impact_score': round(r.overall_impact_score, 2),
                'target_categories': r.priority_categories
            }
            for r in recs[:10]
        ]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_category_rankings(
        self,
        team: TeamProjection,
        all_teams: List[TeamProjection]
    ) -> Dict[str, float]:
        """Calculate team's ranking in each category across the league."""
        rankings = {}

        for cat in self.categories:
            team_val = team.projected_stats.get(cat, 0)

            # Count how many teams are better
            better_count = sum(
                1 for t in all_teams
                if t.team_id != team.team_id and
                (t.projected_stats.get(cat, 0) > team_val if cat not in self.negative_categories
                 else t.projected_stats.get(cat, 0) < team_val)
            )

            # Rank is 1-indexed
            rankings[cat] = better_count + 1

        return rankings

    def _get_current_standing(
        self,
        team_id: int,
        standings: Dict[int, Tuple[int, int, int]],
        all_team_ids: List[int]
    ) -> int:
        """Get current standing for a team based on record."""
        sorted_teams = sorted(
            all_team_ids,
            key=lambda tid: (
                standings.get(tid, (0, 0, 0))[0],  # Wins
                -standings.get(tid, (0, 0, 0))[1],  # Losses
                standings.get(tid, (0, 0, 0))[2]   # Ties
            ),
            reverse=True
        )

        for rank, tid in enumerate(sorted_teams, 1):
            if tid == team_id:
                return rank

        return len(all_team_ids)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_matchup_analyzer(
    categories: Optional[List[str]] = None,
    num_simulations: int = DEFAULT_SIMULATIONS
) -> MatchupAnalyzer:
    """
    Factory function to create a configured matchup analyzer.

    Args:
        categories: Custom category list
        num_simulations: Number of Monte Carlo simulations

    Returns:
        Configured MatchupAnalyzer
    """
    return MatchupAnalyzer(
        categories=categories,
        num_simulations=num_simulations
    )


def quick_matchup_analysis(
    team1_stats: Dict[str, float],
    team2_stats: Dict[str, float],
    team1_name: str = "Your Team",
    team2_name: str = "Opponent"
) -> Dict[str, Any]:
    """
    Quick matchup analysis between two teams.

    Args:
        team1_stats: Team 1 per-game projections
        team2_stats: Team 2 per-game projections
        team1_name: Team 1 name
        team2_name: Team 2 name

    Returns:
        Analysis results dictionary
    """
    analyzer = MatchupAnalyzer(num_simulations=5000)

    team1 = TeamProjection(
        team_id=1,
        team_name=team1_name,
        projected_stats=team1_stats
    )

    team2 = TeamProjection(
        team_id=2,
        team_name=team2_name,
        projected_stats=team2_stats
    )

    cat_probs, win_prob, result = analyzer.analyze_matchup(team1, team2)

    return {
        'team1': team1_name,
        'team2': team2_name,
        'category_win_probabilities': {k: round(v, 3) for k, v in cat_probs.items()},
        'overall_win_probability': round(win_prob, 3),
        'projected_result': result.result_string,
        'expected_category_wins': round(sum(cat_probs.values()), 1),
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Demo/test entry point for matchup analysis."""
    logger.info("=" * 60)
    logger.info("H2H Matchup Analyzer Demo")
    logger.info("=" * 60)

    # Create analyzer
    analyzer = MatchupAnalyzer(num_simulations=10000)

    # Example teams
    team1 = TeamProjection(
        team_id=1,
        team_name="My Team",
        projected_stats={
            'pts': 115.0,
            'trb': 45.0,
            'ast': 28.0,
            'stl': 8.5,
            'blk': 5.5,
            '3p': 14.0,
            'fg_pct': 0.475,
            'ft_pct': 0.820,
            'tov': 12.5
        }
    )

    team2 = TeamProjection(
        team_id=2,
        team_name="Opponent",
        projected_stats={
            'pts': 112.0,
            'trb': 48.0,
            'ast': 25.0,
            'stl': 7.0,
            'blk': 6.5,
            '3p': 12.5,
            'fg_pct': 0.465,
            'ft_pct': 0.780,
            'tov': 11.0
        }
    )

    # Analyze matchup
    print("\n" + "=" * 60)
    print(f"Matchup: {team1.team_name} vs {team2.team_name}")
    print("=" * 60)

    cat_probs, win_prob, result = analyzer.analyze_matchup(team1, team2)

    print(f"\nOverall Win Probability: {win_prob:.1%}")
    print(f"Projected Result: {result.result_string}")

    print("\nCategory Win Probabilities:")
    print("-" * 40)
    print(f"{'Category':<12} {'Win Prob':<12} {'Outlook':<15}")
    print("-" * 40)

    for cat in STANDARD_CATEGORIES:
        prob = cat_probs.get(cat, 0.5)
        if prob >= 0.65:
            outlook = "Strong Win"
        elif prob >= 0.55:
            outlook = "Likely Win"
        elif prob >= 0.45:
            outlook = "Toss-up"
        elif prob >= 0.35:
            outlook = "Likely Loss"
        else:
            outlook = "Strong Loss"

        print(f"{cat.upper():<12} {prob:>8.1%}    {outlook:<15}")

    # Weekly projection
    print("\n" + "=" * 60)
    print("Weekly Matchup Projection")
    print("=" * 60)

    weekly = analyzer.project_weekly_matchup(team1, team2, week=12)

    print(f"\nExpected Category Wins: {weekly.expected_category_wins:.1f}")
    print(f"Matchup Win Probability: {weekly.matchup_win_prob:.1%}")
    print(f"Projected Result: {weekly.projected_result}")

    if weekly.punt_categories:
        print(f"\nPunt Categories: {', '.join(c.upper() for c in weekly.punt_categories)}")

    # Season simulation
    print("\n" + "=" * 60)
    print("Season Simulation (Demo)")
    print("=" * 60)

    # Create a small league for demo
    teams = [
        team1, team2,
        TeamProjection(team_id=3, team_name="Team C", projected_stats={
            'pts': 108.0, 'trb': 42.0, 'ast': 26.0, 'stl': 7.5, 'blk': 5.0,
            '3p': 11.0, 'fg_pct': 0.455, 'ft_pct': 0.790, 'tov': 13.0
        }),
        TeamProjection(team_id=4, team_name="Team D", projected_stats={
            'pts': 118.0, 'trb': 40.0, 'ast': 30.0, 'stl': 9.0, 'blk': 4.0,
            '3p': 15.5, 'fg_pct': 0.485, 'ft_pct': 0.850, 'tov': 14.0
        }),
    ]

    # Simple round-robin schedule
    schedule = {
        1: [(13, 2), (14, 3), (15, 4)],
        2: [(13, 1), (14, 4), (15, 3)],
        3: [(13, 4), (14, 1), (15, 2)],
        4: [(13, 3), (14, 2), (15, 1)],
    }

    current_standings = {
        1: (7, 5, 0),
        2: (6, 6, 0),
        3: (5, 7, 0),
        4: (8, 4, 0),
    }

    results = analyzer.simulate_season(
        teams=teams,
        schedule=schedule,
        current_standings=current_standings,
        current_week=13,
        total_weeks=15,
        playoff_teams=2,
        num_simulations=5000
    )

    print("\nProjected Final Standings:")
    print("-" * 60)
    print(f"{'#':<3} {'Team':<15} {'Proj Record':<15} {'Playoff %':<12} {'Champ %':<10}")
    print("-" * 60)

    for i, res in enumerate(results, 1):
        proj_rec = f"{res.projected_wins:.1f}-{res.projected_losses:.1f}-{res.projected_ties:.1f}"
        print(f"{i:<3} {res.team_name:<15} {proj_rec:<15} {res.playoff_probability*100:>8.1f}%    {res.championship_probability*100:>6.1f}%")

    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())

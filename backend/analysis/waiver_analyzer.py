"""
Waiver Wire Analyzer for Fantasy Basketball Optimizer.

Uses z-score based value calculations to analyze waiver wire moves:
- Add/drop analysis with net z-score impact
- Best drop candidate suggestions
- Category impact breakdown
- Recommendations (ADD/PASS/CONSIDER)

Reuses TradeAnalyzer logic since add/drop is essentially a 1-for-1 trade.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from backend.analysis.trade_analyzer import TradeAnalyzer, TradePlayer

logger = logging.getLogger(__name__)

# Categories where lower is better
REVERSE_CATEGORIES = {'to', 'turnovers', 'TO'}


@dataclass
class WaiverAnalysis:
    """Complete analysis of a waiver wire add/drop move."""
    # Players involved
    add_player_name: str
    add_player_id: int
    add_player_z_score: float
    drop_player_name: str
    drop_player_id: int
    drop_player_z_score: float

    # Z-score impact
    net_z_score_change: float  # Per-game z-score gain/loss
    total_value_change: float  # Total z-score change (per_game × games)

    # Category impact
    category_changes: Dict[str, float]  # Per-game impact per category
    improves_categories: List[str]  # Categories that improve
    hurts_categories: List[str]  # Categories that get worse

    # Recommendation
    recommendation: str  # 'ADD', 'PASS', 'CONSIDER'
    grade: str  # 'A+', 'A', 'B+', 'B', 'C', 'D', 'F'
    reason: str  # Human-readable explanation

    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'add_player_name': self.add_player_name,
            'add_player_id': self.add_player_id,
            'add_player_z_score': round(self.add_player_z_score, 3),
            'drop_player_name': self.drop_player_name,
            'drop_player_id': self.drop_player_id,
            'drop_player_z_score': round(self.drop_player_z_score, 3),
            'net_z_score_change': round(self.net_z_score_change, 3),
            'total_value_change': round(self.total_value_change, 2),
            'category_changes': {k: round(v, 2) for k, v in self.category_changes.items()},
            'improves_categories': self.improves_categories,
            'hurts_categories': self.hurts_categories,
            'recommendation': self.recommendation,
            'grade': self.grade,
            'reason': self.reason,
            'analyzed_at': self.analyzed_at.isoformat(),
        }


@dataclass
class DropSuggestion:
    """Suggested player to drop from roster."""
    player_name: str
    player_id: int
    player_z_score: float
    position: str
    impact_if_dropped: str  # Description of what you lose

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'player_name': self.player_name,
            'player_id': self.player_id,
            'player_z_score': round(self.player_z_score, 3),
            'position': self.position,
            'impact_if_dropped': self.impact_if_dropped,
        }


@dataclass
class WaiverRecommendation:
    """Complete waiver wire recommendation with add and drop."""
    player_to_add: Dict[str, Any]  # Full player data
    analysis: WaiverAnalysis
    rank: int  # 1 = best pickup

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'player_to_add': self.player_to_add,
            'analysis': self.analysis.to_dict(),
            'rank': self.rank,
        }


class WaiverAnalyzer:
    """
    Analyzes waiver wire moves using z-score based valuation.

    Uses the same z-score value system as TradeAnalyzer for consistent
    player valuation across all features.
    """

    def __init__(
        self,
        league_averages: Optional[Dict[str, Dict[str, float]]] = None,
        categories: Optional[List[str]] = None,
        num_teams: int = 10
    ):
        """
        Initialize WaiverAnalyzer.

        Args:
            league_averages: Dict of stat_key -> {'mean': float, 'std': float}
            categories: List of scoring category keys from league settings
            num_teams: Number of teams in league
        """
        self.league_averages = league_averages or {}
        self.num_teams = num_teams

        # Categories must come from league settings
        if not categories or len(categories) == 0:
            logger.warning("No categories provided to WaiverAnalyzer")
            self.categories = []
        else:
            self.categories = [c.lower() if isinstance(c, str) else c for c in categories]
            logger.info(f"WaiverAnalyzer initialized with {len(self.categories)} categories")

        # Create internal TradeAnalyzer for reusing analysis logic
        self._trade_analyzer = TradeAnalyzer(
            league_averages=league_averages,
            categories=categories,
            num_teams=num_teams
        )

    def analyze_add_drop(
        self,
        player_to_add: Dict[str, Any],
        current_roster: List[Dict[str, Any]],
        player_to_drop: Optional[Dict[str, Any]] = None,
    ) -> WaiverAnalysis:
        """
        Analyze adding a waiver player and dropping someone from roster.

        If player_to_drop not specified, finds the best drop candidate.

        Args:
            player_to_add: Free agent player data (must have per_game_stats)
            current_roster: List of current roster players
            player_to_drop: Optional specific player to drop

        Returns:
            WaiverAnalysis with z-score based breakdown
        """
        logger.info(f"=== WAIVER ANALYSIS: Add {player_to_add.get('name', 'Unknown')} ===")

        # If no drop specified, find best candidate
        if player_to_drop is None:
            drop_suggestion = self.suggest_best_drop(current_roster)
            if drop_suggestion is None:
                return self._create_no_drop_analysis(player_to_add)
            player_to_drop = self._find_player_by_id(current_roster, drop_suggestion.player_id)

        if player_to_drop is None:
            return self._create_no_drop_analysis(player_to_add)

        logger.info(f"Drop candidate: {player_to_drop.get('name', 'Unknown')}")

        # Create TradePlayer objects
        trade_player_out = self._trade_analyzer.create_trade_player(
            player_to_drop, self.league_averages
        )
        trade_player_in = self._trade_analyzer.create_trade_player(
            player_to_add, self.league_averages
        )

        # Run trade analysis (add/drop is a 1-for-1 trade)
        trade_analysis = self._trade_analyzer.analyze_trade(
            players_out=[trade_player_out],
            players_in=[trade_player_in],
        )

        # Map trade analysis to waiver analysis
        return WaiverAnalysis(
            add_player_name=player_to_add.get('name', 'Unknown'),
            add_player_id=player_to_add.get('player_id', 0),
            add_player_z_score=trade_player_in.z_score_value,
            drop_player_name=player_to_drop.get('name', 'Unknown'),
            drop_player_id=player_to_drop.get('player_id', 0),
            drop_player_z_score=trade_player_out.z_score_value,
            net_z_score_change=trade_analysis.net_z_score_change,
            total_value_change=trade_analysis.total_value_change,
            category_changes=trade_analysis.category_changes,
            improves_categories=self._get_improves(trade_analysis.category_changes),
            hurts_categories=self._get_hurts(trade_analysis.category_changes),
            recommendation=self._generate_recommendation(trade_analysis.net_z_score_change),
            grade=trade_analysis.trade_grade,
            reason=self._generate_reason(
                player_to_add.get('name', 'Unknown'),
                player_to_drop.get('name', 'Unknown'),
                trade_analysis.net_z_score_change,
                trade_analysis.category_strengths,
                trade_analysis.category_weaknesses,
            ),
        )

    def suggest_best_drop(
        self,
        current_roster: List[Dict[str, Any]],
        exclude_player_ids: Optional[List[int]] = None,
    ) -> Optional[DropSuggestion]:
        """
        Find the lowest-impact player to drop from roster.

        Returns the player with lowest z-score value who is droppable.

        Args:
            current_roster: List of roster players
            exclude_player_ids: Player IDs to exclude from consideration

        Returns:
            DropSuggestion for worst player, or None if no droppable players
        """
        exclude_ids = set(exclude_player_ids or [])

        # Filter to droppable players only
        droppable = []
        for player in current_roster:
            player_id = player.get('player_id', 0)

            # Skip excluded players
            if player_id in exclude_ids:
                continue

            # Check if player is droppable (ESPN flag)
            if not player.get('droppable', True):
                logger.debug(f"Skipping non-droppable player: {player.get('name')}")
                continue

            # Skip IR slot players (they don't take an active roster spot)
            lineup_slot = player.get('lineup_slot_id') or player.get('lineupSlotId')
            if lineup_slot == 13:  # IR slot
                logger.debug(f"Skipping IR player: {player.get('name')}")
                continue

            droppable.append(player)

        if not droppable:
            logger.warning("No droppable players on roster")
            return None

        # Calculate z-scores if not already present
        for player in droppable:
            if 'z_score_value' not in player and 'per_game_value' not in player:
                trade_player = self._trade_analyzer.create_trade_player(
                    player, self.league_averages
                )
                player['z_score_value'] = trade_player.z_score_value

        # Sort by z-score (lowest first = worst player)
        droppable.sort(key=lambda p: p.get('z_score_value') or p.get('per_game_value', 0))

        worst_player = droppable[0]
        z_score = worst_player.get('z_score_value') or worst_player.get('per_game_value', 0)

        logger.info(
            f"Suggested drop: {worst_player.get('name')} "
            f"(z={z_score:.2f})"
        )

        # Generate impact description
        impact = self._describe_drop_impact(worst_player)

        return DropSuggestion(
            player_name=worst_player.get('name', 'Unknown'),
            player_id=worst_player.get('player_id', 0),
            player_z_score=z_score,
            position=worst_player.get('position', 'UTIL'),
            impact_if_dropped=impact,
        )

    def find_best_pickups(
        self,
        available_players: List[Dict[str, Any]],
        current_roster: List[Dict[str, Any]],
        weak_categories: Optional[List[str]] = None,
        projected_category_ranks: Optional[Dict[str, float]] = None,
        max_suggestions: int = 10,
    ) -> List[WaiverRecommendation]:
        """
        Find best waiver wire pickups using TWO-TIER LOGIC:

        1. If ANY category is in bottom half (rank > num_teams/2):
           → Prioritize pickups that improve ANY bottom-half category

        2. If ALL categories are in top half:
           → Prioritize pickups that improve ONLY the weakest category

        Args:
            available_players: Free agents to consider
            current_roster: Current roster players
            weak_categories: Legacy param - categories to prioritize (overrides auto-detection)
            projected_category_ranks: Dict of {category: rank OR roto_points} for auto-detection
            max_suggestions: Maximum recommendations to return

        Returns:
            List of WaiverRecommendation sorted by value
        """
        logger.info("=" * 60)
        logger.info(f"=== FIND_BEST_PICKUPS DEBUG ===")
        logger.info(f"[DEBUG] available_players count: {len(available_players)}")
        logger.info(f"[DEBUG] current_roster count: {len(current_roster)}")
        logger.info(f"[DEBUG] weak_categories (param): {weak_categories}")
        logger.info(f"[DEBUG] projected_category_ranks (param): {projected_category_ranks}")
        logger.info(f"[DEBUG] max_suggestions: {max_suggestions}")
        logger.info(f"[DEBUG] self.num_teams: {self.num_teams}")
        logger.info(f"[DEBUG] self.categories: {self.categories}")
        logger.info(f"[DEBUG] self.league_averages keys: {list(self.league_averages.keys()) if self.league_averages else 'EMPTY'}")
        logger.info("=" * 60)

        # Determine target categories using two-tier logic
        target_categories = weak_categories or []

        if not target_categories and projected_category_ranks:
            target_categories = self._identify_target_categories(
                projected_category_ranks,
                self.num_teams
            )

        if target_categories:
            logger.info(f"[DEBUG] Target categories for pickups: {target_categories}")
        else:
            logger.info("[DEBUG] No target categories - ranking by overall z-score improvement")

        recommendations = []

        # Get best drop candidate once
        logger.info("[DEBUG] Calling suggest_best_drop...")
        best_drop = self.suggest_best_drop(current_roster)
        if best_drop is None:
            logger.warning("[DEBUG] No droppable players - cannot make recommendations")
            return []

        logger.info(f"[DEBUG] Best drop candidate: {best_drop.player_name} (z={best_drop.player_z_score:.3f})")
        drop_player = self._find_player_by_id(current_roster, best_drop.player_id)
        logger.info(f"[DEBUG] drop_player found: {drop_player is not None}")

        # Score each available player
        logger.info(f"[DEBUG] Starting to analyze {len(available_players)} available players...")
        analyzed_count = 0
        added_count = 0
        error_count = 0

        for player in available_players:
            try:
                analyzed_count += 1
                if analyzed_count <= 3:
                    logger.info(f"[DEBUG] Analyzing player {analyzed_count}: {player.get('name', 'Unknown')}")
                    logger.info(f"[DEBUG]   player keys: {list(player.keys())}")
                    logger.info(f"[DEBUG]   per_game_stats: {player.get('per_game_stats', 'NOT_PRESENT')}")

                analysis = self.analyze_add_drop(
                    player_to_add=player,
                    current_roster=current_roster,
                    player_to_drop=drop_player,
                )

                if analyzed_count <= 3:
                    logger.info(f"[DEBUG]   net_z_score_change: {analysis.net_z_score_change:.3f}")
                    logger.info(f"[DEBUG]   recommendation: {analysis.recommendation}")

                # Only suggest if net positive (or slight negative if improves target cats)
                if analysis.net_z_score_change > -0.5:
                    # Bonus for improving target categories
                    target_cat_bonus = 0
                    if target_categories:
                        for cat in analysis.improves_categories:
                            if cat.lower() in [tc.lower() for tc in target_categories]:
                                target_cat_bonus += 0.5

                    recommendations.append({
                        'player': player,
                        'analysis': analysis,
                        'score': analysis.net_z_score_change + target_cat_bonus,
                    })
                    added_count += 1

            except Exception as e:
                error_count += 1
                logger.warning(f"[DEBUG] Error analyzing player {player.get('name')}: {e}")
                import traceback
                if error_count <= 3:
                    logger.warning(f"[DEBUG] Traceback: {traceback.format_exc()}")
                continue

        logger.info(f"[DEBUG] Analysis complete: analyzed={analyzed_count}, added={added_count}, errors={error_count}")

        # Sort by score (highest first)
        recommendations.sort(key=lambda r: r['score'], reverse=True)

        # Convert to WaiverRecommendation objects
        result = []
        for i, rec in enumerate(recommendations[:max_suggestions]):
            result.append(WaiverRecommendation(
                player_to_add=rec['player'],
                analysis=rec['analysis'],
                rank=i + 1,
            ))

        logger.info(f"Generated {len(result)} waiver recommendations")
        return result

    def _identify_target_categories(
        self,
        category_ranks: Dict[str, float],
        num_teams: int
    ) -> List[str]:
        """
        Identify target categories using TWO-TIER LOGIC:

        1. If ANY category is in bottom half (rank > num_teams/2):
           → Target ALL bottom-half categories

        2. If ALL categories are in top half:
           → Target ONLY the weakest category (or all tied for weakest)

        Args:
            category_ranks: Dict of {category: rank OR roto_points}
            num_teams: Number of teams in the league

        Returns:
            List of category names to target
        """
        if not category_ranks:
            return []

        # Detect if values are Roto points vs ranks
        max_value = max(category_ranks.values()) if category_ranks else 0
        values_are_roto_points = max_value == num_teams or max_value == num_teams - 0.5

        # Convert to ranks if needed
        actual_ranks = {}
        for cat, value in category_ranks.items():
            if values_are_roto_points:
                rank = num_teams - value + 1
            else:
                rank = value
            actual_ranks[cat] = rank

        logger.info(f"Category ranks for waiver targeting: {actual_ranks}")

        # Bottom half threshold
        bottom_half_threshold = num_teams / 2

        # Find categories in bottom half
        bottom_half_categories = {
            cat: rank for cat, rank in actual_ranks.items()
            if rank > bottom_half_threshold
        }

        # CASE 1: Bottom-half categories exist
        if bottom_half_categories:
            target_categories = list(bottom_half_categories.keys())
            target_categories.sort(key=lambda c: actual_ranks.get(c, 0), reverse=True)
            logger.info(f"WAIVER CASE 1: Targeting {len(target_categories)} bottom-half categories")
            return target_categories

        # CASE 2: All in top half - target weakest only
        weakest_rank = max(actual_ranks.values())
        target_categories = [
            cat for cat, rank in actual_ranks.items()
            if rank == weakest_rank
        ]
        logger.info(f"WAIVER CASE 2: All top-half, targeting weakest (rank {weakest_rank}): {target_categories}")
        return target_categories

    def get_drop_candidates(
        self,
        current_roster: List[Dict[str, Any]],
        limit: int = 5,
    ) -> List[DropSuggestion]:
        """
        Get ranked list of drop candidates from roster.

        Args:
            current_roster: Current roster players
            limit: Maximum candidates to return

        Returns:
            List of DropSuggestion sorted by z-score (worst first)
        """
        candidates = []
        exclude_ids = []

        for _ in range(limit):
            suggestion = self.suggest_best_drop(
                current_roster,
                exclude_player_ids=exclude_ids,
            )
            if suggestion is None:
                break
            candidates.append(suggestion)
            exclude_ids.append(suggestion.player_id)

        return candidates

    def _generate_recommendation(self, net_z_change: float) -> str:
        """Generate ADD/PASS/CONSIDER recommendation."""
        if net_z_change >= 1.0:
            return 'ADD'
        elif net_z_change >= 0.3:
            return 'CONSIDER'
        elif net_z_change >= -0.3:
            return 'CONSIDER'  # Borderline, might be worth for category fit
        else:
            return 'PASS'

    def _generate_reason(
        self,
        add_name: str,
        drop_name: str,
        net_z: float,
        strengths: List[str],
        weaknesses: List[str],
    ) -> str:
        """Generate human-readable explanation."""
        strengths_str = ', '.join(strengths[:3]).upper() if strengths else 'overall value'
        weaknesses_str = ', '.join(weaknesses[:3]).upper() if weaknesses else 'some categories'

        if net_z >= 1.5:
            return f"Excellent pickup. {add_name} is significantly better than {drop_name}. Improves {strengths_str}."
        elif net_z >= 0.5:
            return f"Good pickup. {add_name} improves your roster by {net_z:+.2f} z-score/game. Helps {strengths_str}."
        elif net_z >= 0:
            return f"Slight upgrade. {add_name} marginally better than {drop_name}. Consider if you need {strengths_str}."
        elif net_z >= -0.5:
            return f"Sidegrade. Minor downgrade overall but may help specific categories. Loses {weaknesses_str}."
        else:
            return f"Not recommended. {drop_name} is more valuable than {add_name}. You would lose {weaknesses_str}."

    def _get_improves(self, category_changes: Dict[str, float]) -> List[str]:
        """Get list of categories that improve with this move."""
        improves = []
        for cat, change in category_changes.items():
            cat_lower = cat.lower() if isinstance(cat, str) else str(cat)

            # Handle reverse categories (turnovers)
            if cat_lower in REVERSE_CATEGORIES:
                # For TO, negative change is good
                if change < -0.1:
                    improves.append(cat.upper())
            else:
                # Threshold varies by category type
                threshold = 0.5 if cat_lower in ['fg_pct', 'ft_pct', 'fg%', 'ft%'] else 0.3
                if change > threshold:
                    improves.append(cat.upper())
        return improves

    def _get_hurts(self, category_changes: Dict[str, float]) -> List[str]:
        """Get list of categories that get worse with this move."""
        hurts = []
        for cat, change in category_changes.items():
            cat_lower = cat.lower() if isinstance(cat, str) else str(cat)

            # Handle reverse categories (turnovers)
            if cat_lower in REVERSE_CATEGORIES:
                # For TO, positive change is bad
                if change > 0.1:
                    hurts.append(cat.upper())
            else:
                # Threshold varies by category type
                threshold = 0.5 if cat_lower in ['fg_pct', 'ft_pct', 'fg%', 'ft%'] else 0.3
                if change < -threshold:
                    hurts.append(cat.upper())
        return hurts

    def _describe_drop_impact(self, player: Dict[str, Any]) -> str:
        """Generate description of what dropping a player costs."""
        z_score = player.get('z_score_value') or player.get('per_game_value', 0)
        name = player.get('name', 'Unknown')

        if z_score >= 2.0:
            return f"Losing {name} significantly hurts your team"
        elif z_score >= 1.0:
            return f"Solid contributor, losing {name} has moderate impact"
        elif z_score >= 0:
            return f"Average player, minimal impact if dropped"
        elif z_score >= -1.0:
            return f"Below average, dropping {name} frees roster spot"
        else:
            return f"Weakest roster player, optimal drop candidate"

    def _find_player_by_id(
        self,
        roster: List[Dict[str, Any]],
        player_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Find player in roster by ID."""
        for player in roster:
            if player.get('player_id') == player_id:
                return player
        return None

    def _create_no_drop_analysis(self, player_to_add: Dict[str, Any]) -> WaiverAnalysis:
        """Create analysis when no drop candidate is available."""
        trade_player = self._trade_analyzer.create_trade_player(
            player_to_add, self.league_averages
        )

        return WaiverAnalysis(
            add_player_name=player_to_add.get('name', 'Unknown'),
            add_player_id=player_to_add.get('player_id', 0),
            add_player_z_score=trade_player.z_score_value,
            drop_player_name='No droppable player',
            drop_player_id=0,
            drop_player_z_score=0,
            net_z_score_change=0,
            total_value_change=0,
            category_changes={},
            improves_categories=[],
            hurts_categories=[],
            recommendation='PASS',
            grade='N/A',
            reason='No droppable players available on roster.',
        )

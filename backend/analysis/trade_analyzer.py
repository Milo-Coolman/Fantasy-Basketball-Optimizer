"""
Trade Analyzer for Fantasy Basketball Optimizer.

Uses z-score based value calculations to analyze trade impact on:
- Per-game fantasy value
- Category strengths/weaknesses
- Projected Roto standings
- Overall fairness

Key Features:
- Z-score based player valuation (same system as start limit optimizer)
- Category-by-category impact analysis
- Projected standings simulation with new roster
- Fairness score calculation for both sides
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime

logger = logging.getLogger(__name__)

# Categories where lower is better
REVERSE_CATEGORIES = {'to', 'turnovers', 'TO'}

# Map alternate stat key names to canonical names
STAT_KEY_MAP = {
    # Canonical -> alternates
    'pts': ['pts', 'PTS', 'points'],
    'reb': ['reb', 'REB', 'rebounds'],
    'ast': ['ast', 'AST', 'assists'],
    'stl': ['stl', 'STL', 'steals'],
    'blk': ['blk', 'BLK', 'blocks'],
    '3pm': ['3pm', '3PM', 'threes', 'three_pointers_made'],
    'to': ['to', 'TO', 'turnovers'],
    'fg_pct': ['fg_pct', 'FG%', 'FG_PCT', 'fgPct', 'field_goal_pct'],
    'ft_pct': ['ft_pct', 'FT%', 'FT_PCT', 'ftPct', 'free_throw_pct'],
    # Makes/Attempts for proper percentage calculation
    'fgm': ['fgm', 'FGM', 'field_goals_made'],
    'fga': ['fga', 'FGA', 'field_goals_attempted'],
    'ftm': ['ftm', 'FTM', 'free_throws_made'],
    'fta': ['fta', 'FTA', 'free_throws_attempted'],
}

# Possible field names for z-score value in player data
Z_SCORE_FIELD_NAMES = ['z_score_value', 'z_value', 'zScore', 'z_score', 'per_game_value', 'value']


@dataclass
class TradePlayer:
    """Represents a player involved in a trade."""
    player_id: int
    player_name: str
    nba_team: str
    z_score_value: float  # Per-game z-score value
    per_game_stats: Dict[str, float]  # Per-game averages
    eligible_slots: List[int] = field(default_factory=list)
    projected_games: int = 0  # Remaining games this season
    injury_status: str = 'ACTIVE'

    @property
    def total_value(self) -> float:
        """Total z-score value for rest of season."""
        return self.z_score_value * self.projected_games


@dataclass
class TradeAnalysis:
    """Complete analysis of a proposed trade using z-score comparison."""
    # Z-score impact
    net_z_score_change: float  # Total per-game z-score gain/loss
    total_value_change: float  # Total z-score change (per_game × games)

    # Category impact
    category_changes: Dict[str, float]  # Per-game impact on each stat
    category_z_changes: Dict[str, float]  # Z-score change per category

    # Fairness assessment
    fairness_score: float  # -10 to +10 (negative = you lose, positive = you gain)
    trade_grade: str  # A+, A, B+, B, C, D, F

    # Recommendation
    recommendation: str  # 'ACCEPT', 'REJECT', 'CONSIDER', 'COUNTER'
    reason: str  # Explanation of recommendation

    # Category details
    category_strengths: List[str]  # Categories you improve
    category_weaknesses: List[str]  # Categories you hurt

    # Players involved
    players_out: List[str]  # Names of players you're trading away
    players_in: List[str]  # Names of players you're receiving

    # Multi-player trade support
    additional_drops: List[Dict[str, Any]] = field(default_factory=list)  # Auto-dropped to fit roster
    trade_type: str = "1-for-1"  # e.g., "2-for-1", "2-for-2", "3-for-1"

    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'net_z_score_change': round(self.net_z_score_change, 3),
            'total_value_change': round(self.total_value_change, 2),
            'category_changes': {k: round(v, 2) for k, v in self.category_changes.items()},
            'category_z_changes': {k: round(v, 3) for k, v in self.category_z_changes.items()},
            'fairness_score': round(self.fairness_score, 2),
            'trade_grade': self.trade_grade,
            'recommendation': self.recommendation,
            'reason': self.reason,
            'category_strengths': self.category_strengths,
            'category_weaknesses': self.category_weaknesses,
            'players_out': self.players_out,
            'players_in': self.players_in,
            'additional_drops': self.additional_drops,
            'trade_type': self.trade_type,
            'analyzed_at': self.analyzed_at.isoformat(),
        }


class TradeAnalyzer:
    """
    Analyzes fantasy basketball trades using z-score based valuation.

    Uses the same z-score value system as the start limit optimizer to ensure
    consistent player valuation across all features.
    """

    def __init__(
        self,
        league_averages: Optional[Dict[str, Dict[str, float]]] = None,
        categories: Optional[List[str]] = None,
        num_teams: int = 10
    ):
        """
        Initialize TradeAnalyzer.

        Args:
            league_averages: Dict of stat_key -> {'mean': float, 'std': float}
                            from StartLimitOptimizer.calculate_league_averages()
            categories: List of scoring category keys - REQUIRED from league.scoring_settings
            num_teams: Number of teams in league (for Roto point calculation)
        """
        self.league_averages = league_averages or {}
        self.num_teams = num_teams

        # Categories MUST come from league settings
        if not categories or len(categories) == 0:
            logger.error("NO CATEGORIES PROVIDED! Trade analyzer requires league scoring categories.")
            logger.error("This will result in empty category analysis.")
            self.categories = []
        else:
            # Normalize category names to lowercase
            self.categories = [c.lower() if isinstance(c, str) else c for c in categories]
            logger.info(f"TradeAnalyzer initialized with {len(self.categories)} categories: {self.categories}")

    def analyze_trade(
        self,
        players_out: List[TradePlayer],
        players_in: List[TradePlayer],
        current_roster: Optional[List[Dict]] = None,
        roster_size_limit: Optional[int] = None,
    ) -> TradeAnalysis:
        """
        Analyze a proposed trade using z-score comparison.

        Supports multi-player trades (1-for-1, 2-for-1, 2-for-2, 3-for-1, etc.)
        and automatically determines additional drops needed to fit roster limits.

        Args:
            players_out: Players you're trading away
            players_in: Players you're receiving
            current_roster: Full roster to check roster limits (optional)
            roster_size_limit: Maximum roster size (optional, default 15)

        Returns:
            TradeAnalysis with z-score based breakdown including additional_drops
        """
        # Determine trade type
        trade_type = f"{len(players_out)}-for-{len(players_in)}"

        logger.info(f"=== TRADE ANALYSIS START ===")
        logger.info(f"Trade type: {trade_type}")
        logger.info(f"Players out: {len(players_out)}, Players in: {len(players_in)}")
        logger.info(f"Categories being analyzed: {self.categories}")

        # Calculate additional drops needed for roster management
        additional_drops = []
        additional_drop_players = []

        if current_roster and roster_size_limit:
            additional_drops, additional_drop_players = self._calculate_additional_drops(
                players_out=players_out,
                players_in=players_in,
                current_roster=current_roster,
                roster_size_limit=roster_size_limit,
            )

        # 1. Calculate z-score impact
        logger.info("--- Players OUT (you're giving away) ---")
        z_out = 0.0
        for p in players_out:
            logger.info(f"  {p.player_name}: z_score={p.z_score_value:+.3f}, games={p.projected_games}")
            z_out += p.z_score_value

        logger.info("--- Players IN (you're receiving) ---")
        z_in = 0.0
        for p in players_in:
            logger.info(f"  {p.player_name}: z_score={p.z_score_value:+.3f}, games={p.projected_games}")
            z_in += p.z_score_value

        # Include additional drops in z-score calculation
        z_dropped = 0.0
        if additional_drop_players:
            logger.info("--- Additional DROPS (to fit roster) ---")
            for p in additional_drop_players:
                logger.info(f"  {p.player_name}: z_score={p.z_score_value:+.3f} (auto-drop)")
                z_dropped += p.z_score_value

        # Net change: what you get - (what you give + what you drop)
        net_z_change = z_in - (z_out + z_dropped)

        # Total value (z-score × projected games)
        total_out = sum(p.total_value for p in players_out)
        total_in = sum(p.total_value for p in players_in)
        total_dropped = sum(p.total_value for p in additional_drop_players)
        total_value_change = total_in - (total_out + total_dropped)

        logger.info(f"--- Z-SCORE SUMMARY ---")
        logger.info(f"  Players OUT total z-score: {z_out:+.3f}")
        logger.info(f"  Players IN total z-score:  {z_in:+.3f}")
        if z_dropped != 0:
            logger.info(f"  Additional drops z-score:  {z_dropped:+.3f}")
        logger.info(f"  NET Z-SCORE CHANGE:        {net_z_change:+.3f} (positive = you gain)")
        logger.info(f"  Total value change:        {total_value_change:+.2f}")

        # 2. Calculate per-category impact (include additional drops as losses)
        all_players_out = list(players_out) + additional_drop_players
        category_changes = self._calculate_category_changes(all_players_out, players_in)
        category_z_changes = self._calculate_category_z_changes(all_players_out, players_in)

        # 3. Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        logger.info("--- IDENTIFYING STRENGTHS/WEAKNESSES ---")
        for cat, change in category_changes.items():
            # Skip if change is negligible
            threshold = 0.5 if cat in ['fg_pct', 'ft_pct'] else 0.1
            if abs(change) < threshold:
                continue

            # Handle reverse categories (turnovers) - negative change = good
            is_reverse = cat in REVERSE_CATEGORIES

            if is_reverse:
                if change < 0:
                    strengths.append(cat)
                else:
                    weaknesses.append(cat)
            else:
                if change > 0:
                    strengths.append(cat)
                else:
                    weaknesses.append(cat)

        # 4. Calculate fairness score and grade based on z-score
        fairness_score = self._calculate_fairness(net_z_change, total_value_change)
        trade_grade = self._calculate_grade(net_z_change)

        # 5. Generate recommendation based on z-score
        recommendation, reason = self._generate_recommendation(
            net_z_change=net_z_change,
            total_value_change=total_value_change,
            fairness_score=fairness_score,
            strengths=strengths,
            weaknesses=weaknesses,
            players_out=players_out,
            players_in=players_in,
            additional_drops=additional_drops
        )

        return TradeAnalysis(
            net_z_score_change=net_z_change,
            total_value_change=total_value_change,
            category_changes=category_changes,
            category_z_changes=category_z_changes,
            fairness_score=fairness_score,
            trade_grade=trade_grade,
            recommendation=recommendation,
            reason=reason,
            category_strengths=strengths,
            category_weaknesses=weaknesses,
            players_out=[p.player_name for p in players_out],
            players_in=[p.player_name for p in players_in],
            additional_drops=additional_drops,
            trade_type=trade_type,
        )

    def _get_stat_value(self, per_game_stats: Dict[str, float], category: str) -> float:
        """
        Get stat value from per_game_stats, checking alternate key names.

        Args:
            per_game_stats: Player's per-game stats dict
            category: Canonical category name (e.g., 'fg_pct')

        Returns:
            Stat value or 0.0 if not found
        """
        # Try canonical name first
        if category in per_game_stats:
            return float(per_game_stats[category] or 0)

        # Try alternate names
        alternates = STAT_KEY_MAP.get(category, [category])
        for alt in alternates:
            if alt in per_game_stats:
                return float(per_game_stats[alt] or 0)

        return 0.0

    def _calculate_category_changes(
        self,
        players_out: List[TradePlayer],
        players_in: List[TradePlayer]
    ) -> Dict[str, float]:
        """Calculate per-game stat changes for each category."""
        changes = {}

        logger.info("--- CATEGORY CHANGES ---")

        for cat in self.categories:
            cat_lower = cat.lower() if isinstance(cat, str) else cat

            # Handle percentage categories specially - calculate from makes/attempts
            if cat_lower in ['fg%', 'fg_pct', 'fgpct']:
                out_fgm = sum(self._get_stat_value(p.per_game_stats, 'fgm') for p in players_out)
                out_fga = sum(self._get_stat_value(p.per_game_stats, 'fga') for p in players_out)
                in_fgm = sum(self._get_stat_value(p.per_game_stats, 'fgm') for p in players_in)
                in_fga = sum(self._get_stat_value(p.per_game_stats, 'fga') for p in players_in)

                out_pct = out_fgm / out_fga if out_fga > 0 else 0
                in_pct = in_fgm / in_fga if in_fga > 0 else 0
                change = (in_pct - out_pct) * 100  # Convert to percentage points

                logger.info(f"  FG%: OUT={out_pct:.3f} ({out_fgm:.1f}/{out_fga:.1f}), "
                           f"IN={in_pct:.3f} ({in_fgm:.1f}/{in_fga:.1f}), change={change:+.2f}%")

            elif cat_lower in ['ft%', 'ft_pct', 'ftpct']:
                out_ftm = sum(self._get_stat_value(p.per_game_stats, 'ftm') for p in players_out)
                out_fta = sum(self._get_stat_value(p.per_game_stats, 'fta') for p in players_out)
                in_ftm = sum(self._get_stat_value(p.per_game_stats, 'ftm') for p in players_in)
                in_fta = sum(self._get_stat_value(p.per_game_stats, 'fta') for p in players_in)

                out_pct = out_ftm / out_fta if out_fta > 0 else 0
                in_pct = in_ftm / in_fta if in_fta > 0 else 0
                change = (in_pct - out_pct) * 100  # Convert to percentage points

                logger.info(f"  FT%: OUT={out_pct:.3f} ({out_ftm:.1f}/{out_fta:.1f}), "
                           f"IN={in_pct:.3f} ({in_ftm:.1f}/{in_fta:.1f}), change={change:+.2f}%")

            else:
                # Regular counting stats - sum the values
                out_total = sum(self._get_stat_value(p.per_game_stats, cat) for p in players_out)
                in_total = sum(self._get_stat_value(p.per_game_stats, cat) for p in players_in)
                change = in_total - out_total

                direction = "+" if change > 0 else ""
                logger.info(f"  {cat}: OUT={out_total:.2f}, IN={in_total:.2f}, change={direction}{change:.2f}")

            changes[cat] = change

        return changes

    def _calculate_category_z_changes(
        self,
        players_out: List[TradePlayer],
        players_in: List[TradePlayer]
    ) -> Dict[str, float]:
        """Calculate z-score changes for each category."""
        changes = {}

        if not self.league_averages:
            logger.warning("No league_averages provided for z-score calculation")
            return {cat: 0.0 for cat in self.categories}

        logger.info("--- CATEGORY Z-SCORE CHANGES ---")

        for cat in self.categories:
            cat_lower = cat.lower() if isinstance(cat, str) else cat
            avg_data = self.league_averages.get(cat, {'mean': 0, 'std': 1})
            mean = avg_data.get('mean', 0)
            std = avg_data.get('std', 1) or 1  # Avoid division by zero

            # Handle percentage categories - calculate from makes/attempts
            if cat_lower in ['fg%', 'fg_pct', 'fgpct']:
                out_fgm = sum(self._get_stat_value(p.per_game_stats, 'fgm') for p in players_out)
                out_fga = sum(self._get_stat_value(p.per_game_stats, 'fga') for p in players_out)
                in_fgm = sum(self._get_stat_value(p.per_game_stats, 'fgm') for p in players_in)
                in_fga = sum(self._get_stat_value(p.per_game_stats, 'fga') for p in players_in)

                out_pct = (out_fgm / out_fga * 100) if out_fga > 0 else mean
                in_pct = (in_fgm / in_fga * 100) if in_fga > 0 else mean

                out_z = (out_pct - mean) / std if std > 0 else 0
                in_z = (in_pct - mean) / std if std > 0 else 0

            elif cat_lower in ['ft%', 'ft_pct', 'ftpct']:
                out_ftm = sum(self._get_stat_value(p.per_game_stats, 'ftm') for p in players_out)
                out_fta = sum(self._get_stat_value(p.per_game_stats, 'fta') for p in players_out)
                in_ftm = sum(self._get_stat_value(p.per_game_stats, 'ftm') for p in players_in)
                in_fta = sum(self._get_stat_value(p.per_game_stats, 'fta') for p in players_in)

                out_pct = (out_ftm / out_fta * 100) if out_fta > 0 else mean
                in_pct = (in_ftm / in_fta * 100) if in_fta > 0 else mean

                out_z = (out_pct - mean) / std if std > 0 else 0
                in_z = (in_pct - mean) / std if std > 0 else 0

            else:
                # Regular counting stats
                out_total = sum(self._get_stat_value(p.per_game_stats, cat) for p in players_out)
                in_total = sum(self._get_stat_value(p.per_game_stats, cat) for p in players_in)

                out_z = (out_total - mean * len(players_out)) / std if std > 0 else 0
                in_z = (in_total - mean * len(players_in)) / std if std > 0 else 0

            z_change = in_z - out_z

            # Flip sign for reverse categories
            if cat in REVERSE_CATEGORIES:
                z_change = -z_change

            changes[cat] = z_change
            logger.debug(f"  {cat}: out_z={out_z:.3f}, in_z={in_z:.3f}, change={z_change:+.3f}")

        return changes

    def _calculate_additional_drops(
        self,
        players_out: List[TradePlayer],
        players_in: List[TradePlayer],
        current_roster: List[Dict],
        roster_size_limit: int,
    ) -> Tuple[List[Dict[str, Any]], List[TradePlayer]]:
        """
        Calculate additional drops needed to fit roster after trade.

        For multi-player trades where you receive more players than you give,
        this determines which players need to be dropped to stay under the limit.
        Drops the LOWEST z-score players first (worst players = best drop candidates).

        Args:
            players_out: Players being traded away
            players_in: Players being received
            current_roster: Full current roster
            roster_size_limit: Maximum roster size

        Returns:
            Tuple of (additional_drops as dicts, additional_drop_players as TradePlayer)
        """
        # Calculate net player change
        net_player_change = len(players_in) - len(players_out)
        current_roster_size = len(current_roster)
        post_trade_size = current_roster_size + net_player_change

        logger.info(f"--- ROSTER MANAGEMENT ---")
        logger.info(f"  Current roster size: {current_roster_size}")
        logger.info(f"  Net player change: {net_player_change:+d}")
        logger.info(f"  Post-trade size: {post_trade_size}")
        logger.info(f"  Roster limit: {roster_size_limit}")

        # Check if additional drops are needed
        additional_drops_needed = max(0, post_trade_size - roster_size_limit)

        if additional_drops_needed == 0:
            logger.info("  No additional drops needed")
            return [], []

        logger.info(f"  ⚠️ Need to drop {additional_drops_needed} additional player(s) to fit roster")

        # Get IDs of players being traded away (don't consider them for dropping)
        players_out_ids = {p.player_id for p in players_out}

        # Find droppable players from current roster
        droppable_candidates = []
        logger.info(f"--- ANALYZING ROSTER FOR DROP CANDIDATES ---")

        for player in current_roster:
            player_id = player.get('player_id', 0)
            player_name = player.get('name', 'Unknown')

            # Skip players being traded away
            if player_id in players_out_ids:
                logger.debug(f"  Skipping {player_name}: being traded away")
                continue

            # Skip non-droppable players (ESPN flag)
            if not player.get('droppable', True):
                logger.info(f"  Skipping {player_name}: marked non-droppable")
                continue

            # Skip IR slot players
            lineup_slot = player.get('lineup_slot_id') or player.get('lineupSlotId', 0)
            if lineup_slot == 13:  # IR slot
                logger.info(f"  Skipping {player_name}: IR slot")
                continue

            # Get z-score value - check multiple possible field names
            z_score = None
            for field_name in Z_SCORE_FIELD_NAMES:
                if field_name in player and player[field_name] is not None:
                    z_score = float(player[field_name])
                    break

            # Log what we found
            logger.info(f"  Candidate: {player_name} | z_score_value={player.get('z_score_value', 'MISSING')} | "
                       f"per_game_value={player.get('per_game_value', 'MISSING')} | resolved_z={z_score}")

            # Store the resolved z_score for sorting
            player['_resolved_z_score'] = z_score if z_score is not None else float('-inf')
            droppable_candidates.append(player)

        logger.info(f"  Found {len(droppable_candidates)} droppable candidates")

        # Sort by z-score ASCENDING (lowest/worst first = best drop candidates)
        # Using _resolved_z_score to avoid issues with None/0 confusion
        droppable_candidates.sort(key=lambda p: p.get('_resolved_z_score', float('-inf')))

        # Log sorted order
        logger.info(f"--- SORTED DROP CANDIDATES (worst first) ---")
        for i, p in enumerate(droppable_candidates):
            z = p.get('_resolved_z_score', 0)
            logger.info(f"  {i+1}. {p.get('name')}: z={z:+.2f}")

        # Select worst N players to drop
        players_to_drop = droppable_candidates[:additional_drops_needed]

        if len(players_to_drop) < additional_drops_needed:
            logger.error(f"  Not enough droppable players! Need {additional_drops_needed}, found {len(players_to_drop)}")

        # Convert to output formats
        additional_drops = []
        additional_drop_players = []

        logger.info(f"--- SELECTED FOR DROP ---")
        for player in players_to_drop:
            z_score = player.get('_resolved_z_score', 0)

            additional_drops.append({
                'name': player.get('name', 'Unknown'),
                'player_id': player.get('player_id', 0),
                'z_score_value': round(z_score, 2),
                'position': player.get('position', 'UTIL'),
            })

            # Create TradePlayer for category calculations
            trade_player = self.create_trade_player(player, self.league_averages)
            additional_drop_players.append(trade_player)

            logger.info(f"  ✓ Auto-drop: {player.get('name')} (z={z_score:+.2f}) - LOWEST z-score")

        # Clean up temporary field
        for player in droppable_candidates:
            player.pop('_resolved_z_score', None)

        return additional_drops, additional_drop_players

    def _calculate_fairness(
        self,
        net_z_change: float,
        total_value_change: float
    ) -> float:
        """
        Calculate trade fairness score on -10 to +10 scale.

        Combines per-game z-score change and total season value.
        Positive = you win the trade, negative = you lose.
        """
        # Weight per-game change heavily (2x) since it's more stable
        # Total value matters but is more speculative
        raw_score = (net_z_change * 2) + (total_value_change / 50)

        # Clamp to -10 to +10 range
        return max(-10, min(10, raw_score))

    def _calculate_grade(self, net_z_change: float) -> str:
        """Convert z-score change to letter grade."""
        if net_z_change >= 2.0:
            return 'A+'
        elif net_z_change >= 1.5:
            return 'A'
        elif net_z_change >= 1.0:
            return 'B+'
        elif net_z_change >= 0.5:
            return 'B'
        elif net_z_change >= -0.5:
            return 'C'
        elif net_z_change >= -1.0:
            return 'D'
        else:
            return 'F'

    def _generate_recommendation(
        self,
        net_z_change: float,
        total_value_change: float,
        fairness_score: float,
        strengths: List[str],
        weaknesses: List[str],
        players_out: List[TradePlayer],
        players_in: List[TradePlayer],
        additional_drops: Optional[List[Dict]] = None
    ) -> Tuple[str, str]:
        """Generate trade recommendation based on z-score analysis."""

        strengths_str = ', '.join(strengths[:3]).upper() if strengths else 'overall depth'
        weaknesses_str = ', '.join(weaknesses[:3]).upper() if weaknesses else 'overall value'

        # Build trade type description
        trade_type = f"{len(players_out)}-for-{len(players_in)}"
        if additional_drops:
            trade_type += f" (+ {len(additional_drops)} drop)"

        # Strong accept: significant positive z-score
        if net_z_change >= 1.5:
            recommendation = 'ACCEPT'
            reason = (f"Strong accept ({trade_type}). You gain {net_z_change:+.2f} z-score/game. "
                     f"Strengthens: {strengths_str}.")

        # Good accept: positive z-score
        elif net_z_change >= 0.5:
            recommendation = 'ACCEPT'
            reason = (f"Good trade ({trade_type}). You gain {net_z_change:+.2f} z-score/game. "
                     f"Helps: {strengths_str}.")

        # Consider: close to even
        elif net_z_change >= -0.5:
            recommendation = 'CONSIDER'
            if strengths and weaknesses:
                reason = (f"Fair trade ({trade_type}, {net_z_change:+.2f} z-score/game). "
                         f"Gain {', '.join(strengths[:2]).upper()}, "
                         f"lose {', '.join(weaknesses[:2]).upper()}. "
                         f"Accept if it fits your strategy.")
            else:
                reason = f"Even trade ({trade_type}, {net_z_change:+.2f} z-score/game). Accept based on roster needs."

        # Counter: slightly negative
        elif net_z_change >= -1.5:
            recommendation = 'COUNTER'
            reason = (f"Slightly favors your partner ({trade_type}, {net_z_change:+.2f} z-score/game). "
                     f"Ask for more value or a different player.")

        # Reject: significantly negative
        else:
            recommendation = 'REJECT'
            reason = (f"Bad trade ({trade_type}). You lose {abs(net_z_change):.2f} z-score/game. "
                     f"Weakens: {weaknesses_str}.")

        # Add note about additional drops if present
        if additional_drops:
            drop_names = [d['name'] for d in additional_drops]
            reason += f" Note: Must also drop {', '.join(drop_names)} to fit roster."

        return recommendation, reason

    def create_trade_player(
        self,
        player_data: Dict,
        league_averages: Optional[Dict[str, Dict[str, float]]] = None
    ) -> TradePlayer:
        """
        Create a TradePlayer from raw player data.

        Args:
            player_data: Dict with player info (from ESPN API or roster)
            league_averages: League averages for z-score calculation

        Returns:
            TradePlayer with calculated z-score value
        """
        per_game_stats = player_data.get('per_game_stats', {}).copy()  # Copy to avoid mutating original
        player_name = player_data.get('name', 'Unknown')

        # Calculate percentages from makes/attempts if available
        if 'fgm' in per_game_stats and 'fga' in per_game_stats:
            if per_game_stats['fga'] > 0:
                per_game_stats['fg_pct'] = per_game_stats['fgm'] / per_game_stats['fga']
            else:
                per_game_stats['fg_pct'] = 0.0

        if 'ftm' in per_game_stats and 'fta' in per_game_stats:
            if per_game_stats['fta'] > 0:
                per_game_stats['ft_pct'] = per_game_stats['ftm'] / per_game_stats['fta']
            else:
                per_game_stats['ft_pct'] = 0.0

        logger.info(f"=== Creating TradePlayer: {player_name} ===")
        logger.info(f"  player_data keys: {list(player_data.keys())}")
        logger.info(f"  per_game_stats: {per_game_stats}")

        # First, check if z-score is already provided in player_data
        z_score = None
        for field_name in Z_SCORE_FIELD_NAMES:
            if field_name in player_data and player_data[field_name] is not None:
                z_score = float(player_data[field_name])
                logger.info(f"  Found pre-calculated z-score in '{field_name}': {z_score:+.3f}")
                break

        # If not found, calculate from per_game_stats
        if z_score is None:
            if league_averages and len(league_averages) > 0:
                z_score = self._calculate_z_score(per_game_stats, league_averages, player_name)
                logger.info(f"  Calculated z-score from stats: {z_score:+.3f}")
            else:
                z_score = 0.0
                logger.warning(f"  No league_averages provided, z-score = 0")

        logger.info(f"  FINAL z_score_value for {player_name}: {z_score:+.3f}")

        return TradePlayer(
            player_id=player_data.get('player_id', 0),
            player_name=player_name,
            nba_team=player_data.get('nba_team', 'UNK'),
            z_score_value=z_score,
            per_game_stats=per_game_stats,
            eligible_slots=player_data.get('eligible_slots', []),
            projected_games=player_data.get('projected_games', 30),
            injury_status=player_data.get('injury_status', 'ACTIVE'),
        )

    def _calculate_z_score(
        self,
        per_game_stats: Dict[str, float],
        league_averages: Dict[str, Dict[str, float]],
        player_name: str = "Unknown"
    ) -> float:
        """Calculate total z-score for a player."""
        total_z = 0.0

        logger.debug(f"Calculating z-score for {player_name}")
        logger.debug(f"  per_game_stats keys: {list(per_game_stats.keys())}")
        logger.debug(f"  league_averages keys: {list(league_averages.keys())}")

        for cat in self.categories:
            avg_data = league_averages.get(cat, {})
            mean = avg_data.get('mean', 0)
            std = avg_data.get('std', 1)

            # Use helper to find stat value
            value = self._get_stat_value(per_game_stats, cat)

            # Scale percentages (if stored as 0.476 instead of 47.6)
            if cat in ['fg_pct', 'ft_pct'] and 0 < value < 1.0:
                value = value * 100

            # Calculate z-score
            if std > 0:
                z = (value - mean) / std
            else:
                z = 0

            # Flip sign for turnovers (fewer is better)
            if cat in REVERSE_CATEGORIES:
                z = -z

            total_z += z
            logger.debug(f"  {cat}: value={value:.2f}, mean={mean:.2f}, std={std:.2f}, z={z:+.3f}")

        logger.debug(f"  TOTAL Z-SCORE for {player_name}: {total_z:+.3f}")
        return total_z

    def compare_players(
        self,
        player1: TradePlayer,
        player2: TradePlayer
    ) -> Dict[str, Any]:
        """
        Compare two players for trade evaluation.

        Returns dict with comparison details.
        """
        z_diff = player1.z_score_value - player2.z_score_value
        total_diff = player1.total_value - player2.total_value

        category_comparison = {}
        for cat in self.categories:
            p1_val = player1.per_game_stats.get(cat, 0)
            p2_val = player2.per_game_stats.get(cat, 0)
            category_comparison[cat] = {
                'player1': p1_val,
                'player2': p2_val,
                'difference': p1_val - p2_val,
                'winner': player1.player_name if p1_val > p2_val else player2.player_name
            }

        return {
            'player1': player1.player_name,
            'player2': player2.player_name,
            'z_score_difference': z_diff,
            'total_value_difference': total_diff,
            'better_player': player1.player_name if z_diff > 0 else player2.player_name,
            'category_comparison': category_comparison,
        }



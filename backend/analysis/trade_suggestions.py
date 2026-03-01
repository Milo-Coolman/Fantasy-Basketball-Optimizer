"""
Trade Suggestions Generator

Generates intelligent trade suggestions based on:
1. User's weak categories (projected rank >= 6)
2. Other teams' strengths in those categories
3. Fair 1-for-1 trade matches based on z-score similarity
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

# Injury statuses that indicate a player is unavailable for trade consideration
UNAVAILABLE_STATUSES = {'OUT', 'SUSPENSION', 'INACTIVE', 'SUSPENDED'}

# How many days until return before filtering out (default: 14 days = 2 weeks)
MAX_DAYS_UNTIL_RETURN = 14


@dataclass
class TradeSuggestion:
    """A single trade suggestion."""
    target_team_id: int
    target_team_name: str
    give_player_id: int
    give_player_name: str
    give_player_z_score: float
    get_player_id: int
    get_player_name: str
    get_player_z_score: float
    net_z_score_change: float
    improves_categories: List[str]  # Weak categories this trade improves
    hurts_categories: List[str]     # Categories that get worse
    fairness_score: float           # 0-100, higher = more fair
    reason: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'target_team_id': self.target_team_id,
            'target_team_name': self.target_team_name,
            # Detailed player info objects
            'give_player_details': {
                'player_id': self.give_player_id,
                'name': self.give_player_name,
                'z_score': round(self.give_player_z_score, 2),
            },
            'get_player_details': {
                'player_id': self.get_player_id,
                'name': self.get_player_name,
                'z_score': round(self.get_player_z_score, 2),
            },
            # Z-score values
            'net_z_score_change': round(self.net_z_score_change, 2),
            'value_gain': round(self.net_z_score_change, 2),  # Alias for frontend compatibility
            # Categories
            'improves_categories': self.improves_categories,
            'hurts_categories': self.hurts_categories,
            'fairness_score': round(self.fairness_score, 1),
            'reason': self.reason,
            # Simple field names for frontend display
            'target_player': self.get_player_name,  # Player you're getting
            'give_player': self.give_player_name,   # Player you're giving away
            'target_team': self.target_team_name,
        }


# Category name mappings (lowercase stat keys)
CATEGORY_TO_STAT = {
    'PTS': 'pts', 'REB': 'reb', 'AST': 'ast', 'STL': 'stl', 'BLK': 'blk',
    '3PM': '3pm', 'FG%': 'fg_pct', 'FT%': 'ft_pct', 'TO': 'to',
    'pts': 'pts', 'reb': 'reb', 'ast': 'ast', 'stl': 'stl', 'blk': 'blk',
    '3pm': '3pm', 'fg_pct': 'fg_pct', 'ft_pct': 'ft_pct', 'to': 'to',
}

# Canonical category names (uppercase display names)
STAT_TO_CATEGORY = {
    'pts': 'PTS', 'reb': 'REB', 'ast': 'AST', 'stl': 'STL', 'blk': 'BLK',
    '3pm': '3PM', 'fg_pct': 'FG%', 'ft_pct': 'FT%', 'to': 'TO',
}

# Unique scoring categories (for iteration without duplicates)
SCORING_CATEGORIES = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FG%', 'FT%', 'TO']


class TradeSuggestionGenerator:
    """
    Generates trade suggestions for a user's team based on category weaknesses.

    Strategy:
    1. Identify user's weak categories (projected rank >= 6 in a 10-team league)
    2. Find teams that are strong (rank <= 3) in those categories
    3. Look for fair 1-for-1 trade matches (similar overall z-score value)
    4. Suggest trades that improve weak categories without hurting too much
    """

    # Trade suggestion mode z-score ranges
    # (min_z, max_z) - trades outside this range are filtered out
    MODE_RANGES = {
        'conservative': (-0.5, 0.5),   # Only very fair trades
        'normal': (-0.25, 1.0),        # Slight advantage OK (default)
        'aggressive': (0.0, 1.5),      # Only trades that benefit you
    }

    def __init__(
        self,
        trade_analyzer: Optional[Any] = None,
        categories: Optional[List[str]] = None,
        trade_suggestion_mode: str = 'normal',
        weak_threshold: int = 6,
        strong_threshold: int = 3,
        max_z_diff: float = 1.5,
    ):
        """
        Initialize the trade suggestion generator.

        Args:
            trade_analyzer: Optional TradeAnalyzer instance for detailed analysis
            categories: List of scoring categories from league settings (REQUIRED)
            trade_suggestion_mode: 'conservative', 'normal', or 'aggressive'
            weak_threshold: Rank threshold for "weak" category (>= this is weak)
            strong_threshold: Rank threshold for "strong" category (<= this is strong)
            max_z_diff: Maximum z-score difference for "fair" trades
        """
        self.trade_analyzer = trade_analyzer
        self.trade_suggestion_mode = trade_suggestion_mode or 'normal'
        self.weak_threshold = weak_threshold
        self.strong_threshold = strong_threshold
        self.max_z_diff = max_z_diff

        logger.info(f"Trade suggestion mode: {self.trade_suggestion_mode} "
                   f"(z-score range: {self.MODE_RANGES.get(self.trade_suggestion_mode, self.MODE_RANGES['normal'])})")

        # Categories MUST come from league settings
        if not categories or len(categories) == 0:
            logger.warning("No categories provided! Using default SCORING_CATEGORIES.")
            logger.warning("This may include categories not in your league (e.g., TO).")
            self.categories = SCORING_CATEGORIES
        else:
            # Keep categories as provided (they'll be mapped to stat keys when needed)
            self.categories = list(categories)
            logger.info(f"TradeSuggestionGenerator initialized with {len(self.categories)} categories: {self.categories}")

    def generate_suggestions(
        self,
        user_team_id: int,
        user_roster: List[Dict],
        all_teams_data: Dict[int, Dict],
        league_averages: Dict[str, Dict],
        max_suggestions: int = 5,
        num_teams: int = 10,
    ) -> List[TradeSuggestion]:
        """
        Generate trade suggestions for the user's team.

        Args:
            user_team_id: The user's team ID
            user_roster: List of player dicts on user's team
            all_teams_data: Dict of {team_id: {name, roster, category_ranks, projected_category_ranks}}
            league_averages: Dict of {stat_key: {mean, std}} for z-score calculation
            max_suggestions: Maximum number of suggestions to return
            num_teams: Number of teams in the league

        Returns:
            List of TradeSuggestion objects, sorted by benefit
        """
        suggestions = []

        # Get user's team data
        user_data = all_teams_data.get(user_team_id)
        if not user_data:
            logger.warning(f"User team {user_team_id} not found in all_teams_data")
            return []

        # 1. Identify user's weak categories
        # Prefer projected ranks, fall back to current ranks
        user_ranks = user_data.get('projected_category_ranks') or user_data.get('category_ranks', {})
        weak_categories = self._identify_weak_categories(user_ranks, num_teams)

        if not weak_categories:
            logger.info(f"No weak categories found for team {user_team_id}")
            return []

        logger.info(f"User weak categories: {weak_categories}")

        # Log availability filtering for user roster
        available_user_count = sum(1 for p in user_roster if self._is_player_available(p))
        unavailable_user_count = len(user_roster) - available_user_count
        if unavailable_user_count > 0:
            logger.info(f"User roster: {len(user_roster)} total, {available_user_count} available "
                       f"({unavailable_user_count} filtered: injured/suspended)")
            for p in user_roster:
                if not self._is_player_available(p):
                    logger.info(f"  Filtered from user roster: {p.get('name', 'Unknown')} "
                               f"(status={p.get('injury_status', 'N/A')})")

        # Check if rosters already have z-scores from optimizer
        has_optimizer_z_scores = any(
            p.get('per_game_value') is not None or p.get('z_score_value') is not None
            for p in user_roster
        )
        if has_optimizer_z_scores:
            logger.info("=== USING Z-SCORES FROM OPTIMIZER (consistent values) ===")
        else:
            logger.info("=== CALCULATING Z-SCORES (optimizer data not provided) ===")

        # 2. Process roster z-scores (uses existing if available, calculates if not)
        user_roster_with_z = self._calculate_roster_z_scores(user_roster, league_averages)

        # Log top/bottom z-score players for debugging
        sorted_by_z = sorted(user_roster_with_z, key=lambda p: p.get('z_score_value', 0), reverse=True)
        logger.info("=== USER ROSTER Z-SCORES (top 5 / bottom 5) ===")
        for p in sorted_by_z[:5]:
            logger.info(f"  {p.get('name', 'Unknown')}: z={p.get('z_score_value', 0):+.2f}")
        logger.info("  ...")
        for p in sorted_by_z[-5:]:
            logger.info(f"  {p.get('name', 'Unknown')}: z={p.get('z_score_value', 0):+.2f}")

        # 3. For each weak category, find trade opportunities
        for weak_cat in weak_categories:
            # Find teams strong in this category
            strong_teams = self._find_teams_strong_in_category(
                weak_cat, all_teams_data, user_team_id
            )

            logger.info(f"Teams strong in {weak_cat}: {[all_teams_data[t]['name'] for t in strong_teams]}")

            # For each strong team, find potential trade matches
            for partner_team_id in strong_teams:
                partner_data = all_teams_data[partner_team_id]
                partner_roster = partner_data.get('roster', [])

                if not partner_roster:
                    continue

                # Calculate z-scores for partner's roster
                partner_roster_with_z = self._calculate_roster_z_scores(
                    partner_roster, league_averages
                )

                # Log partner's top z-score players
                partner_sorted = sorted(partner_roster_with_z, key=lambda p: p.get('z_score_value', 0), reverse=True)
                logger.debug(f"Partner team {partner_data.get('name')} top 3 z-scores:")
                for p in partner_sorted[:3]:
                    logger.debug(f"  {p.get('name', 'Unknown')}: z={p.get('z_score_value', 0):+.2f}")

                # Find fair trade matches
                matches = self._find_trade_matches(
                    user_roster=user_roster_with_z,
                    partner_roster=partner_roster_with_z,
                    weak_category=weak_cat,
                    league_averages=league_averages,
                    partner_team_name=partner_data.get('name', 'Unknown'),
                    partner_team_id=partner_team_id,
                )

                suggestions.extend(matches)

        # 4. Deduplicate (same player pairs)
        suggestions = self._deduplicate_suggestions(suggestions)

        # 5. Sort by net benefit (higher is better) and fairness
        suggestions.sort(
            key=lambda s: (s.net_z_score_change, s.fairness_score),
            reverse=True
        )

        # 6. Return top suggestions
        return suggestions[:max_suggestions]

    def _identify_weak_categories(
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

        This balances broad improvement vs focused optimization.

        Args:
            category_ranks: Dict of {category: rank OR roto_points}
                           If values look like Roto points (1-10), converts to ranks
            num_teams: Number of teams in the league

        Returns:
            List of category names to target for trade suggestions
        """
        if not category_ranks:
            logger.warning("No category ranks provided")
            return []

        # Detect if values are Roto points (1-10 scale where higher is better)
        # vs ranks (1-10 scale where lower is better)
        # Heuristic: if max value equals num_teams, likely Roto points
        max_value = max(category_ranks.values()) if category_ranks else 0
        values_are_roto_points = max_value == num_teams or max_value == num_teams - 0.5

        # Convert to ranks if needed (rank = num_teams - roto_points + 1)
        actual_ranks = {}
        for cat, value in category_ranks.items():
            if values_are_roto_points:
                # Convert Roto points to rank
                # 10 points (1st place) -> rank 1
                # 1 point (last place) -> rank 10
                rank = num_teams - value + 1
            else:
                rank = value
            actual_ranks[cat] = rank

        logger.info(f"Category ranks (converted={values_are_roto_points}): {actual_ranks}")

        # Bottom half threshold: rank > num_teams/2
        # For 10-team: rank > 5 (i.e., ranks 6-10 are bottom half)
        bottom_half_threshold = num_teams / 2

        logger.info(f"Bottom half threshold: rank > {bottom_half_threshold}")

        # Find categories in bottom half
        bottom_half_categories = {
            cat: rank for cat, rank in actual_ranks.items()
            if rank > bottom_half_threshold
        }

        # CASE 1: At least one category in bottom half
        # → Target ALL bottom-half categories
        if bottom_half_categories:
            target_categories = list(bottom_half_categories.keys())
            # Sort by weakness (worst first)
            target_categories.sort(key=lambda c: actual_ranks.get(c, 0), reverse=True)
            logger.info(f"CASE 1: Bottom-half categories found: {target_categories}")
            logger.info(f"Generating suggestions for ALL {len(target_categories)} bottom-half categories")
            return target_categories

        # CASE 2: All categories in top half
        # → Target ONLY the weakest category (or all tied for weakest)
        weakest_rank = max(actual_ranks.values())
        target_categories = [
            cat for cat, rank in actual_ranks.items()
            if rank == weakest_rank
        ]

        logger.info(f"CASE 2: All categories in top half!")
        logger.info(f"Targeting weakest (rank {weakest_rank}): {target_categories}")

        return target_categories

    def _find_teams_strong_in_category(
        self,
        category: str,
        all_teams_data: Dict[int, Dict],
        exclude_team_id: int
    ) -> List[int]:
        """Find teams ranked in top 3 for this category."""
        strong_teams = []

        for team_id, data in all_teams_data.items():
            if team_id == exclude_team_id:
                continue

            # Check both projected and current ranks
            ranks = data.get('projected_category_ranks') or data.get('category_ranks', {})
            rank = ranks.get(category, 10)

            if rank <= self.strong_threshold:
                strong_teams.append(team_id)

        return strong_teams

    def _calculate_roster_z_scores(
        self,
        roster: List[Dict],
        league_averages: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Calculate z-score value for each player in roster.

        IMPORTANT: Uses existing z-scores from optimizer/dashboard if available.
        Only recalculates if z-score is missing (ensures consistency with optimizer).
        """
        result = []

        for player in roster:
            player_copy = dict(player)
            player_name = player.get('name', 'Unknown')

            # CHECK FOR EXISTING Z-SCORE FROM OPTIMIZER/DASHBOARD
            # Priority: per_game_value (from optimizer) > z_score_value
            existing_z = player.get('per_game_value') or player.get('z_score_value')

            if existing_z is not None:
                # USE EXISTING Z-SCORE - ensures consistency with optimizer
                player_copy['z_score_value'] = existing_z
                logger.debug(f"Using existing z-score for {player_name}: {existing_z:+.2f}")

                # Still need per_game_stats for category comparison
                per_game = player.get('per_game_stats') or player.get('avg', {})
                if not per_game:
                    stats = player.get('stats', {})
                    for key in ['2026_total', '2025_total', 'total']:
                        if key in stats and 'avg' in stats[key]:
                            per_game = stats[key]['avg']
                            break
                player_copy['per_game_stats'] = per_game

                # Build category z-scores for category comparison (needed for improves/hurts)
                category_z_scores = player.get('category_z_scores', {})
                if not category_z_scores and per_game and league_averages:
                    # Only calculate category z-scores, total already exists
                    for cat in self.categories:
                        stat_key = CATEGORY_TO_STAT.get(cat, cat.lower())
                        avg_data = league_averages.get(stat_key, {})
                        mean = avg_data.get('mean', 0)
                        std = avg_data.get('std', 1) or 1
                        stat_value = self._get_stat_value(per_game, stat_key)

                        # Scale percentages if stored as decimals (0.476 -> 47.6)
                        stat_key_lower = stat_key.lower()
                        if stat_key_lower in ['fg_pct', 'ft_pct', 'fg%', 'ft%', '3p_pct']:
                            if stat_value > 0 and stat_value < 1:
                                stat_value = stat_value * 100
                            if mean > 0 and mean < 1:
                                mean = mean * 100
                                std = std * 100 if std < 1 else std

                        z = (stat_value - mean) / std
                        if stat_key_lower in ['to', 'tov', 'turnovers']:
                            z = -z
                        category_z_scores[stat_key] = z
                player_copy['category_z_scores'] = category_z_scores
                result.append(player_copy)
                continue

            # NO EXISTING Z-SCORE - Calculate from scratch
            logger.debug(f"Calculating z-score for {player_name} (not found in optimizer data)")

            # Get per-game stats
            per_game = player.get('per_game_stats') or player.get('avg', {})
            if not per_game:
                # Try to extract from stats
                stats = player.get('stats', {})
                for key in ['2026_total', '2025_total', 'total']:
                    if key in stats and 'avg' in stats[key]:
                        per_game = stats[key]['avg']
                        break

            # Calculate total z-score using LEAGUE CATEGORIES ONLY
            total_z = 0.0
            category_z_scores = {}

            for cat in self.categories:
                # Map category name to stat key
                stat_key = CATEGORY_TO_STAT.get(cat, cat.lower())

                avg_data = league_averages.get(stat_key, {})
                mean = avg_data.get('mean', 0)
                std = avg_data.get('std', 1)

                if std == 0:
                    std = 1

                # Get player's stat value
                stat_value = self._get_stat_value(per_game, stat_key)

                # Scale percentages if stored as decimals (0.476 -> 47.6)
                stat_key_lower = stat_key.lower()
                if stat_key_lower in ['fg_pct', 'ft_pct', 'fg%', 'ft%', '3p_pct']:
                    if stat_value > 0 and stat_value < 1:
                        stat_value = stat_value * 100
                    # Also check if mean looks like it's in decimal form
                    if mean > 0 and mean < 1:
                        mean = mean * 100
                        std = std * 100 if std < 1 else std

                # Calculate z-score
                z = (stat_value - mean) / std

                # Flip sign for turnovers (lower is better)
                if stat_key_lower in ['to', 'tov', 'turnovers']:
                    z = -z

                category_z_scores[stat_key] = z
                total_z += z

            player_copy['z_score_value'] = total_z
            player_copy['category_z_scores'] = category_z_scores
            player_copy['per_game_stats'] = per_game

            # Log calculated z-score for key players
            logger.debug(f"Calculated z_score for {player_name}: {total_z:+.2f} "
                        f"(categories: {len(self.categories)})")

            result.append(player_copy)

        return result

    def _get_stat_value(self, stats: Dict, stat_key: str) -> float:
        """Get a stat value from player stats, handling various key formats."""
        if not stats:
            return 0.0

        # Direct match
        if stat_key in stats:
            return stats[stat_key] or 0.0

        # Try uppercase
        if stat_key.upper() in stats:
            return stats[stat_key.upper()] or 0.0

        # Try lowercase
        if stat_key.lower() in stats:
            return stats[stat_key.lower()] or 0.0

        # Common mappings
        mappings = {
            'pts': ['PTS', 'points'],
            'reb': ['REB', 'rebounds', 'TRB'],
            'ast': ['AST', 'assists'],
            'stl': ['STL', 'steals'],
            'blk': ['BLK', 'blocks'],
            '3pm': ['3PM', '3P', 'threes'],
            'fg_pct': ['FG%', 'FGP', 'fg_percent'],
            'ft_pct': ['FT%', 'FTP', 'ft_percent'],
            'to': ['TO', 'TOV', 'turnovers'],
        }

        key_lower = stat_key.lower()
        if key_lower in mappings:
            for alt in mappings[key_lower]:
                if alt in stats:
                    return stats[alt] or 0.0

        return 0.0

    def _is_player_available(self, player: Dict) -> bool:
        """
        Check if a player is available to play (not out for season or long-term injured).

        Filters out:
        - Players marked as out for season
        - Players with long-term injuries (return date > 14 days away)
        - Players with OUT/SUSPENSION/INACTIVE status

        Args:
            player: Player dictionary with injury_status, injury_details, etc.

        Returns:
            True if player is available or will return soon, False otherwise
        """
        player_name = player.get('name', 'Unknown')

        # Check injury_details for out_for_season flag
        injury_details = player.get('injury_details') or {}
        if injury_details.get('out_for_season', False):
            logger.debug(f"Trade filter: {player_name} - OUT FOR SEASON")
            return False

        # Check expected return date
        expected_return = player.get('expected_return_date')
        if not expected_return and injury_details:
            expected_return = injury_details.get('expected_return_date')

        if expected_return:
            # Handle both date objects and strings
            if isinstance(expected_return, str):
                try:
                    expected_return = datetime.strptime(expected_return, '%Y-%m-%d').date()
                except ValueError:
                    expected_return = None

            if isinstance(expected_return, date):
                today = date.today()
                days_until_return = (expected_return - today).days
                if days_until_return > MAX_DAYS_UNTIL_RETURN:
                    logger.debug(f"Trade filter: {player_name} - returns in {days_until_return} days")
                    return False

        # Check injury status (OUT, SUSPENSION, INACTIVE are not available)
        injury_status = (player.get('injury_status') or '').upper()
        if injury_status in UNAVAILABLE_STATUSES:
            logger.debug(f"Trade filter: {player_name} - status={injury_status}")
            return False

        # Player is available (ACTIVE, DTD, QUESTIONABLE, or no status)
        return True

    def _get_category_z_score(
        self,
        player: Dict,
        category: str,
        league_averages: Dict
    ) -> float:
        """Get z-score for a specific category."""
        player_name = player.get('name') or player.get('player_name') or 'Unknown'
        cat_lower = category.lower()
        is_pct = cat_lower in ['fg_pct', 'ft_pct', 'fg%', 'ft%', '3p_pct']

        # Check if pre-calculated z-scores exist
        if 'category_z_scores' in player:
            # Try various key formats
            z = player['category_z_scores'].get(category)
            if z is None:
                z = player['category_z_scores'].get(category.lower())
            if z is None:
                z = player['category_z_scores'].get(category.upper())
            if z is not None:
                if is_pct:
                    logger.debug(f"[PCT Z-SCORE] {player_name} {category}: using pre-calc z={z:+.3f}")
                return z

        # Calculate on the fly
        per_game = player.get('per_game_stats', {})

        # DEBUG: Log player data structure for percentage stats
        if is_pct:
            logger.info(f"[PCT DEBUG] {player_name} - checking data structure:")
            logger.info(f"  Player keys: {list(player.keys())}")
            logger.info(f"  per_game_stats exists: {'per_game_stats' in player}")
            logger.info(f"  per_game_stats keys: {list(per_game.keys()) if per_game else 'EMPTY'}")
            # Check if stats are nested elsewhere
            if 'stats' in player:
                logger.info(f"  player['stats'] keys: {list(player['stats'].keys()) if isinstance(player['stats'], dict) else type(player['stats'])}")
            # Try to find FG%/FT% anywhere in the player dict
            for key in ['fg_pct', 'ft_pct', 'FG%', 'FT%', 'FGP', 'FTP']:
                if key in per_game:
                    logger.info(f"  FOUND in per_game_stats: {key} = {per_game[key]}")
                if key in player:
                    logger.info(f"  FOUND in player root: {key} = {player[key]}")

        stat_value = self._get_stat_value(per_game, category)

        # Also try getting stat from player root if per_game failed
        if stat_value == 0 and is_pct:
            # Try getting from player root
            direct_value = player.get(category) or player.get(category.lower()) or player.get(category.upper())
            if direct_value:
                logger.info(f"[PCT DEBUG] {player_name} - Found {category} in player root: {direct_value}")
                stat_value = direct_value

        avg_data = league_averages.get(category, {})
        mean = avg_data.get('mean', 0)
        std = avg_data.get('std', 1)

        # Also check for category with different key format in league_averages
        if not avg_data and is_pct:
            logger.info(f"[PCT DEBUG] {player_name} - league_averages missing '{category}', available keys: {list(league_averages.keys())}")
            # Try alternate keys
            for alt_key in [category.lower(), category.upper(), 'FG%' if 'fg' in category.lower() else 'FT%']:
                if alt_key in league_averages:
                    avg_data = league_averages[alt_key]
                    mean = avg_data.get('mean', 0)
                    std = avg_data.get('std', 1)
                    logger.info(f"[PCT DEBUG] {player_name} - Using alternate key '{alt_key}': mean={mean}, std={std}")
                    break

        if std == 0:
            std = 1

        # Log original values for percentage stats
        if is_pct:
            logger.info(f"[PCT Z-SCORE] {player_name} {category}: BEFORE scaling - stat={stat_value}, mean={mean}, std={std}")

        # Scale percentages if stored as decimals (0.476 -> 47.6)
        # This ensures consistency with league averages which may be in percentage form
        if is_pct:
            if stat_value > 0 and stat_value < 1:
                stat_value = stat_value * 100
                logger.info(f"[PCT Z-SCORE] {player_name} {category}: scaled stat to {stat_value}")
            # Also check if mean looks like it's in decimal form
            if mean > 0 and mean < 1:
                mean = mean * 100
                std = std * 100 if std < 1 else std
                logger.info(f"[PCT Z-SCORE] {player_name} {category}: scaled mean/std to {mean}/{std}")

        z = (stat_value - mean) / std

        if is_pct:
            logger.info(f"[PCT Z-SCORE] {player_name} {category}: FINAL z=({stat_value} - {mean}) / {std} = {z:+.3f}")

        # Flip for turnovers (lower is better)
        if cat_lower in ['to', 'tov']:
            z = -z

        return z

    def _find_trade_matches(
        self,
        user_roster: List[Dict],
        partner_roster: List[Dict],
        weak_category: str,
        league_averages: Dict,
        partner_team_name: str,
        partner_team_id: int,
    ) -> List[TradeSuggestion]:
        """Find fair 1-for-1 trade matches that improve the weak category."""
        matches = []

        # Debug: Check categories for percentage stats
        logger.info(f"=== _find_trade_matches DEBUG ===")
        logger.info(f"self.categories = {self.categories}")
        logger.info(f"Looking for FG%/FT% in categories:")
        for cat in self.categories:
            cat_stat = CATEGORY_TO_STAT.get(cat, cat.lower())
            cat_display = STAT_TO_CATEGORY.get(cat_stat, cat.upper())
            is_pct = cat_stat in ['fg_pct', 'ft_pct']
            logger.info(f"  '{cat}' -> stat_key='{cat_stat}' -> display='{cat_display}' {'[PERCENTAGE]' if is_pct else ''}")
        logger.info(f"league_averages keys: {list(league_averages.keys())}")

        # Map category name to stat key
        stat_key = CATEGORY_TO_STAT.get(weak_category, weak_category.lower())

        # Find partner players who are strong in the weak category
        # FILTER: Skip unavailable players (out for season, long-term injured, suspended)
        partner_strong_players = []
        partner_filtered_count = 0
        for p in partner_roster:
            # Check if player is available (not injured long-term)
            if not self._is_player_available(p):
                partner_filtered_count += 1
                continue

            cat_z = self._get_category_z_score(p, stat_key, league_averages)
            if cat_z > 0.5:  # Above average in this category
                partner_strong_players.append(p)

        if partner_filtered_count > 0:
            logger.info(f"Filtered {partner_filtered_count} unavailable players from {partner_team_name}")

        if not partner_strong_players:
            return []

        # For each strong partner player, find fair trade matches from user's roster
        for partner_player in partner_strong_players:
            partner_z = partner_player.get('z_score_value', 0)
            partner_cat_z = self._get_category_z_score(partner_player, stat_key, league_averages)

            for my_player in user_roster:
                # FILTER: Don't suggest trading away unavailable players
                # (they have low trade value anyway and the trade wouldn't go through)
                if not self._is_player_available(my_player):
                    continue

                my_z = my_player.get('z_score_value', 0)
                my_cat_z = self._get_category_z_score(my_player, stat_key, league_averages)

                # Check if trade is "fair" (similar overall z-score)
                z_diff = abs(partner_z - my_z)
                if z_diff > self.max_z_diff:
                    continue

                # Check if trade improves weak category
                cat_improvement = partner_cat_z - my_cat_z
                if cat_improvement <= 0:
                    continue  # Doesn't improve weak category

                # Calculate net z-score change
                # Formula: (player receiving z-score) - (player giving away z-score)
                # Example: Get Chet (+1.40) - Give Pritchard (-1.40) = +2.80
                net_z_change = partner_z - my_z

                # Filter by trade suggestion mode
                min_z, max_z = self.MODE_RANGES.get(self.trade_suggestion_mode, self.MODE_RANGES['normal'])
                if net_z_change < min_z or net_z_change > max_z:
                    logger.debug(f"Trade filtered by mode '{self.trade_suggestion_mode}': "
                                f"net_z={net_z_change:+.2f} outside range [{min_z}, {max_z}]")
                    continue

                # Get player names (check multiple field names for compatibility)
                my_player_name = my_player.get('name') or my_player.get('player_name') or 'Unknown'
                partner_player_name = partner_player.get('name') or partner_player.get('player_name') or 'Unknown'

                # Debug logging for z-score calculation
                logger.info(f"Trade match: Give {my_player_name} (z={my_z:+.2f}) "
                           f"for {partner_player_name} (z={partner_z:+.2f})")
                logger.info(f"Net z-score: {partner_z:+.2f} - ({my_z:+.2f}) = {net_z_change:+.2f}")

                # Calculate fairness (closer z-scores = more fair)
                fairness = max(0, 100 - (z_diff * 30))

                # Identify which categories improve/hurt
                # Normalize weak_category to display format for consistent display
                weak_stat_key = CATEGORY_TO_STAT.get(weak_category, weak_category.lower())
                weak_cat_display = STAT_TO_CATEGORY.get(weak_stat_key, weak_category.upper())
                improves = [weak_cat_display]
                hurts = []

                # Iterate over LEAGUE-SPECIFIC categories only (not hardcoded SCORING_CATEGORIES)
                logger.info(f"=== Category analysis for {my_player_name} -> {partner_player_name} ===")
                logger.info(f"League categories: {self.categories}")
                for cat in self.categories:
                    # Convert category to stat key for lookup
                    cat_stat = CATEGORY_TO_STAT.get(cat, cat.lower())
                    # Convert to display format for output
                    cat_display = STAT_TO_CATEGORY.get(cat_stat, cat.upper())

                    if cat_display == weak_cat_display:
                        logger.info(f"  {cat} -> {cat_stat} -> {cat_display}: SKIPPED (weak category)")
                        continue

                    partner_c = self._get_category_z_score(partner_player, cat_stat, league_averages)
                    my_c = self._get_category_z_score(my_player, cat_stat, league_averages)
                    diff = partner_c - my_c

                    # Log ALL categories for debugging
                    is_pct = cat_stat in ['fg_pct', 'ft_pct']
                    threshold = 0.3
                    passes_improve = diff > threshold
                    passes_hurt = diff < -threshold
                    logger.info(f"  {cat} -> {cat_stat} -> {cat_display}: "
                               f"partner_z={partner_c:+.3f}, my_z={my_c:+.3f}, diff={diff:+.3f}, "
                               f"threshold={threshold}, improves={passes_improve}, hurts={passes_hurt}"
                               f"{' [PCT]' if is_pct else ''}")

                    # Use threshold of 0.3 z-score difference to classify
                    if diff > 0.3:
                        improves.append(cat_display)
                        logger.info(f"    -> ADDED TO IMPROVES: {cat_display}")
                    elif diff < -0.3:
                        hurts.append(cat_display)
                        logger.info(f"    -> ADDED TO HURTS: {cat_display}")

                # Generate reason
                if net_z_change > 0:
                    reason = f"Upgrades {weak_category} (+{cat_improvement:.1f} z) while gaining overall value"
                elif net_z_change > -0.5:
                    reason = f"Fair swap that improves {weak_category} (+{cat_improvement:.1f} z)"
                else:
                    reason = f"Slight downgrade overall but fixes {weak_category} weakness"

                # Log final improves/hurts lists
                logger.info(f"=== FINAL TRADE SUMMARY ===")
                logger.info(f"  Give: {my_player_name} (z={my_z:+.2f})")
                logger.info(f"  Get:  {partner_player_name} (z={partner_z:+.2f})")
                logger.info(f"  Net z-change: {net_z_change:+.2f}")
                logger.info(f"  IMPROVES: {improves}")
                logger.info(f"  HURTS: {hurts}")
                logger.info(f"  Has FG%? {'FG%' in improves or 'FG%' in hurts}")
                logger.info(f"  Has FT%? {'FT%' in improves or 'FT%' in hurts}")

                matches.append(TradeSuggestion(
                    target_team_id=partner_team_id,
                    target_team_name=partner_team_name,
                    give_player_id=my_player.get('player_id') or my_player.get('espn_player_id', 0),
                    give_player_name=my_player_name,
                    give_player_z_score=my_z,
                    get_player_id=partner_player.get('player_id') or partner_player.get('espn_player_id', 0),
                    get_player_name=partner_player_name,
                    get_player_z_score=partner_z,
                    net_z_score_change=net_z_change,
                    improves_categories=improves,
                    hurts_categories=hurts,
                    fairness_score=fairness,
                    reason=reason,
                ))

        return matches

    def _deduplicate_suggestions(
        self,
        suggestions: List[TradeSuggestion]
    ) -> List[TradeSuggestion]:
        """Remove duplicate suggestions (same player pair)."""
        seen = set()
        unique = []

        for s in suggestions:
            key = (s.give_player_id, s.get_player_id)
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique


def generate_trade_suggestions(
    user_team_id: int,
    user_roster: List[Dict],
    all_teams_data: Dict[int, Dict],
    league_averages: Dict[str, Dict],
    categories: Optional[List[str]] = None,
    trade_suggestion_mode: str = 'normal',
    max_suggestions: int = 5,
    num_teams: int = 10,
    trade_analyzer: Optional[Any] = None,
) -> List[Dict]:
    """
    Convenience function to generate trade suggestions.

    Args:
        user_team_id: User's team ID
        user_roster: List of player dicts on user's team
        all_teams_data: Dict of {team_id: {name, roster, category_ranks, projected_category_ranks}}
        league_averages: Dict of {stat_key: {mean, std}} for z-score calculation
        categories: List of scoring categories from league.scoring_settings (REQUIRED)
        trade_suggestion_mode: 'conservative', 'normal', or 'aggressive'
        max_suggestions: Maximum number of suggestions to return
        num_teams: Number of teams in the league
        trade_analyzer: Optional TradeAnalyzer instance

    Returns:
        List of suggestion dicts ready for JSON serialization.
    """
    logger.info(f"generate_trade_suggestions called with categories: {categories}")
    logger.info(f"Trade suggestion mode: {trade_suggestion_mode}")

    generator = TradeSuggestionGenerator(
        trade_analyzer=trade_analyzer,
        categories=categories,
        trade_suggestion_mode=trade_suggestion_mode,
    )
    suggestions = generator.generate_suggestions(
        user_team_id=user_team_id,
        user_roster=user_roster,
        all_teams_data=all_teams_data,
        league_averages=league_averages,
        max_suggestions=max_suggestions,
        num_teams=num_teams,
    )

    return [s.to_dict() for s in suggestions]

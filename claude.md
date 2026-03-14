# Claude Code Context - Fantasy Basketball Optimizer

## Project Overview
You are helping build a **Fantasy Basketball Optimizer** web application that integrates with ESPN Fantasy Basketball to provide advanced analytics, projections, and recommendations.

**Key Goal:** Parse ESPN league data to project end-of-season standings, recommend trades, and suggest waiver wire acquisitions using a hybrid ML + statistical approach.

---

## Critical Files to Reference
- **fantasy-basketball-optimizer-PRD.md** - Complete Project Requirements Document with all technical specifications
- **This file (claude.md)** - Quick reference and context

---

## Tech Stack

### Backend
- **Python 3.10+**
- **Flask** - Web framework
- **SQLAlchemy** - ORM
- **Flask-Migrate** - Database migrations (Alembic wrapper)
- **PostgreSQL** (production) / **SQLite** (development)
- **espn-api** - ESPN Fantasy Basketball API wrapper
- **scikit-learn** - Machine learning models
- **pandas/numpy** - Data analysis
- **BeautifulSoup4 + requests** - Web scraping (Basketball Reference)
- **Flask-Login** - Authentication
- **APScheduler** - Scheduled tasks (daily data refresh)

### Frontend
- **React 18+**
- **React Router** - Navigation
- **Axios** - HTTP client
- **Chart.js or Recharts** - Data visualization
- **Tailwind CSS or Material-UI** - Styling
- **React Context API** - State management

### Database
- **SQLite** for local development
- **PostgreSQL** for future production

---

## Project Structure

```
fantasy-basketball-optimizer/
├── backend/
│   ├── app.py                      # Flask application entry point
│   ├── config.py                   # Configuration settings
│   ├── models.py                   # SQLAlchemy database models
│   ├── auth.py                     # Authentication routes and logic
│   ├── api/
│   │   ├── __init__.py
│   │   ├── leagues.py              # League-related endpoints
│   │   ├── projections.py          # Projection endpoints
│   │   ├── trades.py               # Trade analysis endpoints
│   │   └── waivers.py              # Waiver recommendations endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── espn_client.py          # ESPN API wrapper service
│   │   ├── cache_service.py        # Data caching logic
│   │   └── scheduler_service.py    # Daily update scheduler
│   ├── projections/
│   │   ├── __init__.py
│   │   ├── ml_model.py             # Machine learning projection model
│   │   ├── statistical_model.py    # Statistical projection model
│   │   ├── hybrid_engine.py        # Combined projection engine
│   │   └── trained_models/         # Pre-trained model files
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── trade_analyzer.py       # Z-score based trade analysis
│   │   ├── trade_suggestions.py    # Trade suggestion generator
│   │   ├── waiver_recommender.py   # Waiver wire recommendations
│   │   └── matchup_analyzer.py     # H2H matchup projections
│   ├── scrapers/
│   │   ├── __init__.py
│   │   └── basketball_reference.py # Basketball Reference scraper
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py              # Utility functions
│   └── tests/
│       ├── __init__.py
│       ├── test_projections.py
│       ├── test_trades.py
│       └── test_espn_client.py
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── App.js                  # Main application component
│   │   ├── index.js                # Entry point
│   │   ├── components/
│   │   │   ├── Navbar.js
│   │   │   ├── Login.js
│   │   │   ├── Register.js
│   │   │   ├── LeagueSetup.js
│   │   │   ├── LeagueDashboard.js   # Main dashboard with all sections
│   │   │   ├── StandingsTable.js
│   │   │   ├── ProjectionsChart.js
│   │   │   ├── CategoryComparisonChart.js
│   │   │   ├── QuickInsights.js     # Trade opportunities, waiver targets
│   │   │   ├── ProjectionSettings.js
│   │   │   ├── StartLimitsInfo.js   # Roto start limit display
│   │   │   ├── WaiverRecommendations.js
│   │   │   ├── WeeklyMatchups.js    # H2H only
│   │   │   └── PlayerCard.js
│   │   ├── services/
│   │   │   └── api.js              # API service layer
│   │   ├── context/
│   │   │   └── AuthContext.js      # Authentication context
│   │   ├── hooks/
│   │   │   └── useAuth.js          # Custom authentication hook
│   │   └── styles/
│   │       └── App.css
│   ├── package.json
│   └── README.md
├── database/
│   ├── schema.sql                  # Database schema
│   └── migrations/                 # Database migrations
├── docs/
│   ├── API.md                      # API documentation
│   ├── USER_GUIDE.md               # User guide for ESPN cookie setup
│   └── DEPLOYMENT.md               # Deployment instructions
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore
├── claude.md                       # This file - context for Claude Code
├── fantasy-basketball-optimizer-PRD.md  # Complete PRD
└── README.md
```

---

## Key Features to Implement

### 1. User Authentication
- User registration with email/password
- Login/logout with Flask-Login
- Secure password hashing (bcrypt)
- Session management

### 2. ESPN Integration
- Users provide ESPN league URL + ESPN_S2 and SWID cookies
- Fetch league settings, scoring rules, rosters, free agents
- Daily automatic data refresh (3 AM default)
- Manual refresh button

### 3. League Support
- **H2H Categories:** Week-by-week projections, playoff probability
- **Rotisserie:** End-of-season category rankings, win probability

### 4. Projection Engine (Hybrid Approach)

#### Data Sources (3 sources)
1. **ESPN Projections** - Preseason and in-season projections from ESPN Fantasy
2. **Current Season Stats** - Actual performance this season
3. **ML Model** - Machine learning predictions based on player features

**Note:** Previous season stats are not used because the ESPN API does not provide them when connected to the current season.

#### Tiered Weighting System (by Games Played)
| Tier | Games Played | ESPN Proj | Current Season | ML Model |
|------|--------------|-----------|----------------|----------|
| 1    | 0-5 games    | 90%       | 0%             | 10%      |
| 2    | 6-15 games   | 55%       | 35%            | 10%      |
| 3    | 16-35 games  | 15%       | 80%            | 5%       |
| 4    | 35+ games    | 0%        | 100%           | 0%       |

#### 10 Component Stats Output
- **Counting Stats:** PTS, REB, AST, STL, BLK, 3PM
- **Shooting Components:** FGM, FGA, FTM, FTA
- **Calculated:** FG% = FGM ÷ FGA, FT% = FTM ÷ FTA

#### Percentage Stat Calculation
Percentage stats are calculated from **weighted component totals** (not averaged):
```
Example: 50/50 weighting ESPN + Current for FG%
ESPN: 5.2 FGM / 11.0 FGA (47.3%)
Current: 4.8 FGM / 10.5 FGA (45.7%)
Combined FGM = 5.0, Combined FGA = 10.75
Combined FG% = 5.0 ÷ 10.75 = 46.5% (NOT average of 47.3% and 45.7%)
```

### 5. Injury Handling Implementation
- **Data Source:** ESPN's `kona_playercard` endpoint via espn-api library
- **Detection:** Only ESPN-reported injuries affect projections (no phantom injuries)
- **IR Slot:** Players in IR identified by `lineupSlotId == 13` from ESPN API
- **Schedule-Based:** Uses actual NBA team schedules for remaining games (not 82-game estimates)
- **Return Projection:** Uses ESPN's expected return date for IR players
- **IR Drop Optimizer (Simplified):** When IR player returns:
  - Sorts droppable players by z-score value (lowest first)
  - Recommends dropping the player with lowest z-score
  - Calculates net gain: `ir_z_value - drop_z_value`
  - Instant execution (no complex Roto simulations needed)

### 6. Projection Method Settings
Users can configure how game rates are calculated for `projected_games`:

**Adaptive Mode (Default):**
| Games Played | Game Rate |
|--------------|-----------|
| 0-4 games    | 90% (grace period for new/returning players) |
| 5+ games     | Actual rate (GP ÷ Team GP) with 75% floor |

**Flat Rate Mode:**
- User sets a fixed percentage (e.g., 85%)
- Applied uniformly to all healthy players
- Useful for simplified projections or testing

**Key Settings:**
- `projection_method`: "adaptive" or "flat_rate" (stored per league)
- `flat_rate_value`: Decimal (0.0-1.0) for flat rate mode
- 75% floor prevents over-penalizing players with early-season absences

### 7. Day-by-Day Start Limit Optimization (Roto)
For leagues with position start limits (e.g., 82 games per position):
- **Daily Simulation:** For each remaining day, identifies which players have games
- **Z-Score Value System:** Players ranked by sum of z-scores across all categories
- **Starter Assignment:** Assigns highest z-value eligible player to each slot
- **Projected Games Enforcement:** Players can't exceed their `projected_games` limit
- **Limit Enforcement:** Tracks `starts_used` per position, stops when limit reached
- **Conflict Resolution:** Prioritizes higher z-value players for limited slots
- **Output:** Realistic projections based on actual games player will start

### 7.1 Z-Score Value Calculation
The optimizer uses z-scores instead of hardcoded category weights:

**Formula:** `z_score = (player_stat - league_mean) / league_std_dev`

**League Averages:**
- Calculated from all rostered players across all teams in the league
- Includes mean and standard deviation for each scoring category
- Cached and reused for consistent player comparisons

**Per-Game Value:**
```python
per_game_value = sum(z_scores for all categories)
# Example: LeBron = +8.5/game (elite), Buddy Hield = -0.82/game (below average)
```

**Category Handling:**
- **Counting Stats:** PTS, REB, AST, STL, BLK, 3PM (higher = better)
- **Turnovers:** Sign flipped (lower TO = positive z-score)
- **Percentages:** FG% and FT% multiplied by 100 before z-score calculation
  - Converts 0.476 → 47.6 to match counting stat scale

**Benefits Over Hardcoded Weights:**
- League-specific: Adapts to YOUR league's player pool
- Scarcity-aware: Rare categories (BLK, STL) naturally weighted higher
- Fair comparison: 25 PTS and 2.5 BLK compared on same scale
- No manual tuning required

### 8. Trade Analyzer (Z-Score Based)

The trade analyzer uses the same z-score value system as the start limit optimizer for consistent player evaluation.

**Trade Analysis Features:**
- **Input:** Multi-player trades (any size, 2 teams only)
- **Z-Score Comparison:** Calculates net z-score change from user's perspective
  - Formula: `net_z_change = (players_received_z) - (players_given_z)`
  - Positive = trade benefits you, Negative = trade hurts you
- **Category Impact Analysis:** Shows which categories improve/hurt
  - Uses 0.3 z-score threshold for significance
  - Separate display for improvements (green) and concerns (red)
- **Fairness Scoring:** -10 to +10 scale based on value differential
- **Recommendation Engine:** ACCEPT / REJECT / CONSIDER / COUNTER
- **Trade Grading:** A+, A, B+, B, C, D, F based on overall value

**Technical Details:**
- Only analyzes league-specific scoring categories (not hardcoded defaults)
- FG% and FT% calculated from components (FGM/FGA, FTM/FTA)
- Percentage stats scaled to 0-100 before z-score calculation

### 8.1 Trade Suggestions Generator

Auto-generates trade opportunities based on projected category weaknesses.

**Algorithm:**
1. Identifies user's weak categories (projected rank >= 6 in 10-team league)
2. Finds teams strong in those categories (rank <= 3)
3. Searches for fair 1-for-1 trade matches (z-score diff < 1.5)
4. Filters by configurable aggressiveness mode
5. Returns ranked suggestions with category impact breakdown

**Configurable Aggressiveness Modes:**
| Mode | Z-Score Range | Description |
|------|---------------|-------------|
| Conservative | -0.5 to +0.5 | Only very fair, balanced trades |
| Normal | -0.25 to +1.0 | Slightly favorable trades OK (default) |
| Aggressive | 0.0 to +1.5 | Only trades that benefit you |

**Database:**
- `trade_suggestion_mode` field added to League model
- Stored per league, persists user preference

**UI Integration:**
- Inline settings panel within Trade Opportunities section
- Auto-saves on mode change
- Settings accessible via gear icon in section header

### 8.2 Multi-Player Trade Support

Supports complex multi-player trades (2-for-1, 2-for-2, 3-for-1, etc.) with automatic roster management.

**Trade Types Supported:**
| Trade Type | Players Given | Players Received | Roster Impact |
|------------|---------------|------------------|---------------|
| 1-for-1 | 1 | 1 | No change |
| 2-for-1 | 1 | 2 | Need to drop 1 |
| 1-for-2 | 2 | 1 | Open 1 slot |
| 2-for-2 | 2 | 2 | No change |
| 3-for-2 | 2 | 3 | Need to drop 1 |

**Automatic Roster Management:**
When a trade creates a roster overflow (receiving more players than giving):
1. **Calculate Overflow:** `additional_drops = players_received - players_given`
2. **Identify Drop Candidates:** All roster players except:
   - Players involved in the trade
   - Players already being dropped
3. **Sort by Z-Score:** Lowest z-score first (worst players = best drop candidates)
4. **Auto-Select Drops:** Recommend dropping the N lowest z-score players

**Z-Score Resolution:**
Multiple field names are checked for z-score value:
```python
Z_SCORE_FIELD_NAMES = ['z_score_value', 'per_game_value', 'zscore', 'z_value']
```
- Handles both dashboard-calculated and analyzer-calculated z-scores
- Falls back to `-inf` if no z-score found (deprioritizes unknowns)
- Properly handles `0` vs `None` (0 is a valid z-score)

**Dynamic Roster Limit:**
- Fetches active roster size from ESPN (excluding IR slots)
- Caches in `active_roster_limit` field in League model
- IR slots identified by names: 'IR', 'IR+', 'IL', 'IL+'

**API Endpoint:**
```
POST /api/leagues/:id/trades/analyze
Body: {
  "team1_id": 1,
  "team1_players": [123, 456],  // Players given
  "team2_id": 2,
  "team2_players": [789],       // Players received
  "current_roster": [...],      // Full roster data with z-scores
  "roster_size_limit": 13       // Optional, auto-fetched if not provided
}

Response includes:
- "additional_drops": [{ player info with z-scores }]
- "trade_type_badge": "2-for-1"
```

**UI Display:**
- Trade type badge (e.g., "2-for-1") shown in analysis results
- Additional drops warning with player names if roster overflow
- Z-score values displayed for all dropped players

**Key Files:**
- `backend/analysis/trade_analyzer.py` - `_calculate_additional_drops()`
- `backend/api/trades.py` - Dynamic roster limit fetching
- `backend/api/dashboard.py` - `_add_z_scores_to_all_rosters()`
- `frontend/src/components/QuickInsights.js` - TradeAnalyzerModal updates

### 9. Waiver Wire Analyzer (Z-Score Based)

The waiver wire analyzer uses the same z-score system as trade and start limit optimization.

**Core Features:**
- **Click-to-Analyze:** Click any waiver target in dashboard to get full analysis
- **Z-Score Based Add/Drop:** Compares free agent z-score to roster's worst player
- **Net Benefit Calculation:** `net_gain = free_agent_z - worst_roster_player_z`
- **Auto-Drop Suggestion:** Identifies lowest z-score player as optimal drop candidate
- **Category Impact:** Shows which categories improve/hurt from the move

**Availability Filtering:**
All waiver suggestions filter out unavailable players:
- Players marked "out for season" by ESPN
- Long-term injuries (expected return > 14 days away)
- Suspended players (`SUSPENSION` status)
- Inactive players (`INACTIVE` status)
- Only suggests immediately playable pickups

**Recommendation Engine:**
| Net Z-Score | Grade | Recommendation |
|-------------|-------|----------------|
| > +1.5 | A+ | ADD (significant upgrade) |
| +1.0 to +1.5 | A | ADD (clear improvement) |
| +0.5 to +1.0 | B | ADD (moderate gain) |
| 0.0 to +0.5 | C | CONSIDER (marginal) |
| < 0.0 | D/F | PASS (not beneficial) |

**Backend Endpoints:**
- `POST /api/leagues/:id/waivers/analyze` - Analyze specific add/drop
- `GET /api/leagues/:id/waivers/suggestions` - Get ranked suggestions
- `GET /api/leagues/:id/waivers/recommendations` - Legacy endpoint

**Key Files:**
- `backend/analysis/waiver_analyzer.py` - WaiverAnalyzer class
- `backend/api/waivers.py` - API endpoints with availability filtering
- `backend/api/dashboard.py` - Waiver targets in quick insights
- `frontend/src/components/QuickInsights.js` - WaiverTargets with click-to-analyze

### 10. Dashboard
- Current vs projected standings
- Playoff/win probability visualization
- Quick insights (top waiver targets, trade opportunities)
- Category strengths/weaknesses

---

---

## Key Bug Fixes (February 2026)

### Tie Handling in Roto Rankings (NEW)
- **Issue:** Two teams tied in a category would get different ranks (e.g., 5 and 6) instead of sharing
- **Fix:** Updated `calculate_roto_category_ranks` and projected standings to use average ranks for ties
- **Impact:** Teams with identical stats now correctly receive average rank (e.g., tied for 3rd = rank 3.5, 7.5 Roto points each). Works for N-way ties (2, 3, 4+ teams)

### Counting Stats Rounding (NEW)
- **Issue:** STL and BLK projected stats showed decimals (e.g., 142.37) while other counting stats were rounded
- **Fix:** Added `round()` to all counting stats in EOS totals calculation (PTS, REB, AST, STL, BLK, 3PM, TO, FGM, FGA, FTM, FTA)
- **Impact:** All counting stats display as whole numbers; percentages (FG%, FT%) remain as decimals

### FG%/FT% Z-Score Calculation (NEW)
- **Issue:** FG% and FT% showing 0.000 z-scores for all players in trade analyzer
- **Fix:** Added percentage calculation from components (FGM/FGA, FTM/FTA) in `espn_client.py` for both `get_all_rosters()` and `_parse_player()`
- **Impact:** Percentage stats now correctly influence trade analysis and suggestions

### Phantom Injuries Removed
- **Issue:** Players were being marked as injured without ESPN reporting an injury
- **Fix:** Injury status now only comes from ESPN's `kona_playercard` endpoint
- **Impact:** Projections no longer incorrectly reduce games for healthy players

### Projected Games Enforcement
- **Issue:** Start limit optimizer wasn't respecting individual player game limits
- **Fix:** Added `projected_games` tracking per player, enforced in daily simulation
- **Impact:** Prevents over-projecting players who miss games due to rest/minor injuries

### Schedule-Based Calculations
- **Issue:** Remaining games calculated from 82-game estimates, not actual schedules
- **Fix:** Uses NBA team schedule data for accurate remaining games
- **Impact:** More accurate projections, especially late in season

### Game Rate Floor
- **Issue:** Players with low early-season game rates were severely under-projected
- **Fix:** 75% minimum floor for players with 5+ games in adaptive mode
- **Impact:** More reasonable projections for players recovering from early issues

---

## Key Bug Fixes (March 2026)

### Auto-Drop Logic Fix
- **Issue:** Auto-drop in multi-player trades was selecting high-value players (e.g., Scottie Barnes +3.56 z-score) instead of lowest z-score players
- **Root Cause 1:** `p.get('z_score_value') or p.get('per_game_value', 0)` treats 0 as falsy, falling back incorrectly
- **Root Cause 2:** `team_rosters` sent to frontend didn't have `z_score_value` calculated
- **Fix:** Added `_resolved_z_score` field with proper `None` vs `0` handling; added `_add_z_scores_to_all_rosters()` helper in dashboard
- **Impact:** Auto-drop now correctly identifies lowest z-score players as drop candidates

### Hardcoded Roster Limit Fix
- **Issue:** Roster limit was hardcoded to 15 instead of fetched from ESPN
- **Fix:** Added `active_roster_limit` to League model, `get_roster_size_info()` to ESPN client
- **Impact:** Correctly excludes IR slots (IR, IR+, IL, IL+) from active roster count

### Z-Score Field Resolution
- **Issue:** Different parts of the app used different field names for z-scores (`z_score_value`, `per_game_value`, `zscore`)
- **Fix:** Added comprehensive field name checking in `_calculate_additional_drops()`
- **Impact:** Consistent z-score resolution across all player data sources

---

## Recent Major Changes (February 2026)

### Z-Score Value System Implementation
**Changed:** Replaced hardcoded category weights with dynamic z-score calculations

**Why the Change:**
- Hardcoded weights (e.g., PTS=1.0, REB=1.5) don't adapt to league-specific scarcity
- Manual weight tuning was arbitrary and didn't reflect actual category value
- Different leagues have different category distributions

**New Approach:**
- Calculate league-wide ROS per-game averages from all rostered players
- Compute z-scores: `(player_stat - mean) / std_dev` for each category
- Sum z-scores across all categories for total per-game value
- Higher z-value players get priority in lineup slot assignments

**Key Files:**
- `backend/projections/start_limit_optimizer.py`:
  - `calculate_league_averages()` - computes league-wide mean/std
  - `calculate_player_value()` - computes z-score based value

**Percentage Stat Handling:**
- FG% and FT% multiplied by 100 before z-score calculation
- Example: 0.476 → 47.6 to match counting stat scale
- Ensures percentages have comparable z-score impact to counting stats

### IR Drop Optimizer Simplification
**Changed:** Replaced complex Roto simulation with simple z-score comparison

**Previous Approach (Removed):**
- For each drop candidate, simulated entire rest-of-season
- Fetched all teams' stats and projected EOS totals
- Ranked teams in each category to calculate Roto points
- Took 10-30 seconds per IR player

**New Approach:**
- Sort droppable players by z-score value (lowest first)
- Drop the player with the lowest z-score
- Calculate net gain: `ir_z_value - drop_z_value`
- Instant execution

**Key Code Change:**
```python
# Simple z-score based drop decision
candidates.sort(key=lambda p: p.get('per_game_value', 0))
worst_player = candidates[0]  # Lowest z-score = best to drop
```

### Deferred: Category Balancing (Phase 2)
**Status:** Not implemented, deferred for future consideration

**Concept:** Instead of maximizing total z-score, balance categories to avoid punting
**Why Deferred:**
- Z-score approach already provides good baseline optimization
- Category balancing adds complexity without clear benefit
- Users can manually identify weak categories from dashboard

### Trade Analyzer & Suggestions (February 2026)
**Status:** Fully implemented

**Trade Analyzer:**
- Z-score based trade evaluation matching start limit optimizer values
- Category-by-category impact analysis (improves/hurts breakdown)
- Uses league-specific scoring categories only (not hardcoded defaults)
- FG% and FT% properly calculated from FGM/FGA and FTM/FTA components
- Fairness scoring, recommendation engine, and trade grading

**Trade Suggestions Generator:**
- Auto-generates trade opportunities based on projected category weaknesses
- Uses projected ranks from dashboard (not current ranks)
- Configurable aggressiveness modes (conservative/normal/aggressive)
- Inline settings panel integrated into Trade Opportunities section
- Database migration added `trade_suggestion_mode` to League model

**Key Files:**
- `backend/analysis/trade_analyzer.py` - TradeAnalyzer class, TradePlayer dataclass
- `backend/analysis/trade_suggestions.py` - TradeSuggestionGenerator class
- `backend/api/trades.py` - API endpoints for trade analysis and suggestions
- `frontend/src/components/QuickInsights.js` - TradeOpportunities with inline settings

**Deferred Features:**
- Trade projected standings calculation (scrapped - z-score only approach is simpler and faster)
- Multi-team trade packages (too complex for now)
- Keeper league considerations (deferred)

### Stats View Toggle in Standings Tables (February 2026)
**Status:** Implemented

**Feature:**
- Dropdown toggle to switch between "Roto Points" and "Actual Stats" views
- Stats view shows actual stat values (e.g., PTS: 10,593, FG%: 47.6%)
- Preserves color coding from Roto points (green/yellow/red tiers)
- Total column always shows Roto points sum (not stats sum)
- Move column preserved in projected standings
- Tooltips show stat value, rank, and points for context

**Key Files:**
- `frontend/src/components/StandingsTable.js` - viewMode state, formatStat helper
- `frontend/src/styles/App.css` - .view-mode-select, .standings-header-row styles

### Auto-Refresh on Settings Change (February 2026)
**Status:** Implemented

**Feature:**
- Dashboard automatically refreshes when projection settings are saved
- Dashboard automatically refreshes when trade suggestion mode is changed
- Settings panels auto-collapse after successful save
- Loading indicator shown during refresh

**Implementation:**
- `onSettingsChange` callback passed from LeagueDashboard to settings components
- Calls `loadDashboard(true)` to trigger full data refresh
- ProjectionSettings shows "Refreshing projections..." message
- TradeOpportunities settings panel closes and triggers refresh

### Multi-Player Trade Support (March 2026)
**Status:** Implemented

**Feature:**
- Support for any trade size (2-for-1, 3-for-2, etc.)
- Automatic roster overflow detection
- Smart drop recommendations based on z-score
- Trade type badges in UI

**Implementation:**
- `_calculate_additional_drops()` in trade_analyzer.py
- Dynamic roster limit fetching from ESPN
- Z-score resolution across multiple field names
- Frontend display of additional drops warning

**Key Files:**
- `backend/analysis/trade_analyzer.py` - Drop logic
- `backend/api/trades.py` - Roster limit handling
- `backend/api/dashboard.py` - Z-score calculation for all rosters
- `frontend/src/components/QuickInsights.js` - UI updates

### Waiver Wire Analyzer (March 2026)
**Status:** Implemented

**Feature:**
- Click-to-analyze any waiver target
- Z-score based add/drop evaluation
- Availability filtering (excludes injured/suspended)
- Category impact analysis

**Implementation:**
- WaiverAnalyzer class with z-score comparison
- Availability check before suggesting players
- Expected return date filtering (>14 days = unavailable)
- Grade assignment based on net z-score gain

**Key Files:**
- `backend/analysis/waiver_analyzer.py` - WaiverAnalyzer class
- `backend/api/waivers.py` - API endpoints with filtering
- `frontend/src/components/QuickInsights.js` - Click handler

### Player Rankings (Planned - Next Priority)
**Status:** Planned

Universal player rankings within each league using z-score based evaluation of all NBA players.

**Core Concept:**
- League-specific player rankings for ALL NBA players (~450+ players)
- Z-score based using the league's actual scoring categories
- Combines rostered players AND free agents into unified ranking
- Uses league-wide NBA averages (not just rostered players) for fairer comparison

**Data Source:**
- All NBA players (rostered + free agents combined)
- League-wide mean/std calculated from entire player pool
- Only includes players with ≥5 games played
- Uses standard deviation across ALL NBA players for accurate relative rankings

**Features:**
- **Sortable Table:** Click any column to sort
  - First click: Descending (best to worst)
  - Second click: Ascending (worst to best)
  - Default: Total z-score descending
- **Position Filter:** Dropdown (All, PG, SG, SF, PF, C)
- **Search Bar:** Filter by player name or NBA team
- **Player Detail Modal:** Click any player for expanded stats
- **Color-Coded Z-Scores:**
  - Green: z > 1.0 (excellent)
  - Black: 0 < z < 1.0 (above average)
  - Gray: -1.0 < z < 0 (below average)
  - Red: z < -1.0 (poor)
- **Injury Status Indicators:** Visual badges for DTD, O, IR

**Performance:**
- 24-hour cache to avoid recalculating on every load
- Cache key: `player_rankings_{league_id}`
- Background refresh available for future optimization
- Client-side filtering/sorting for responsiveness

**Technical Implementation:**

Backend Endpoint:
```
GET /api/leagues/{id}/player-rankings
```

Returns:
- All players with total z-score and per-category z-scores
- Per-game stats for each player
- League-wide mean/std for each category
- Cache expiration timestamp

Calculation Logic:
1. Fetch all rostered players from all teams
2. Fetch all free agents from ESPN
3. Combine into single player list
4. Filter to players with ≥5 games played
5. Calculate league-wide mean/std for each scoring category
6. Compute z-scores: `(player_stat - mean) / std_dev`
7. Sum category z-scores for total z-score
8. Cache results with 24-hour TTL

Frontend Component:
```
frontend/src/components/PlayerRankings.js
```
- New tab in league dashboard alongside Trade Opportunities, Waiver Targets
- Sortable table with dynamic filtering
- Player detail modal on click

**Key Files (To Be Created):**
- `backend/api/player_rankings.py` - API endpoint
- `backend/analysis/player_ranker.py` - Ranking calculation logic
- `frontend/src/components/PlayerRankings.js` - Main component
- `frontend/src/components/PlayerRankingsTable.js` - Sortable table
- `frontend/src/components/ZScoreCell.js` - Color-coded z-score display

**Use Cases:**
- Identify undervalued players on waivers
- Compare players across all categories at once
- Find category specialists (high z-score in one category)
- Draft preparation and player evaluation
- Quick lookup of any player's league-specific value

**Why This Matters:**
- More comprehensive than ESPN's built-in player rater
- Tailored to YOUR league's specific scoring categories
- Z-scores capture relative value (not absolute stats)
- Category specialists easy to spot
- Unified view of all NBA players, not just free agents

**Design Decisions:**
- Uses ALL NBA players for average calculation (fairer than just rostered)
- League-specific (each league has different categories/z-scores)
- 24-hour cache balances freshness with performance
- Client-side filtering avoids repeated API calls

---

## Database Schema (Key Tables)

### Users
- id, email, password_hash, created_at, last_login

### Leagues
- id, user_id, espn_league_id, season, league_name
- espn_s2_cookie, swid_cookie (encrypted)
- league_type (H2H_CATEGORY, H2H_POINTS, ROTO)
- roster_settings (JSONB), scoring_settings (JSONB)
- last_updated, refresh_schedule
- projection_method ("adaptive" or "flat_rate")
- flat_rate_value (decimal, 0.0-1.0)
- trade_suggestion_mode ("conservative", "normal", "aggressive")
- active_roster_limit (integer, cached from ESPN, excludes IR slots)

### Teams
- id, league_id, espn_team_id, team_name, owner_name
- current_record, current_standing, projected_standing
- win_probability

### Players
- id, espn_player_id, name, team (NBA), position, injury_status

### Rosters
- id, team_id, player_id, acquisition_type, roster_slot

### Player_Stats
- id, player_id, season, games_played, stats (pts, reb, ast, etc.)
- stat_date, source (ESPN, BASKETBALL_REFERENCE)

### Projections
- id, player_id, league_id, season, projection_type (STATISTICAL, ML, HYBRID)
- projected stats, fantasy_value, confidence

### Trade_History
- id, league_id, team1_id, team2_id, players (JSONB)
- value_differential, was_suggested, was_accepted

### Waiver_Recommendations
- id, league_id, team_id, player_id, impact_score
- suggested_drop_player_id, recommendation_date

---

## Development Phases

### Phase 1: Foundation ✓
- ✅ Set up project structure
- ✅ Configure Flask backend with SQLAlchemy
- ✅ Set up SQLite database with schema
- ✅ Configure Flask-Migrate for database migrations
- Create React frontend with basic routing
- Implement user authentication
- Build ESPN cookie setup workflow

### Phase 2: Data Integration ✓
- ✅ Integrate espn-api package
- ✅ Build ESPN client service
- ✅ Implement injury handling (kona_playercard endpoint)
- Implement data caching
- Create Basketball Reference scraper
- Build database CRUD operations
- Implement daily refresh scheduler

### Phase 3: Projection Engine (IN PROGRESS)
- ✅ Build projection method settings (adaptive/flat rate)
- ✅ Implement schedule-based remaining games calculation
- ✅ Add projected_games enforcement in start limit optimizer
- Collect training data
- Train ML models
- Build statistical projection component
- Implement hybrid engine
- Create H2H matchup simulator
- Build Roto category predictor

### Phase 4: Core Features
- Build dashboard with standings/projections
- Implement trade analyzer
- Create waiver wire recommender
- Build H2H matchup analysis

### Phase 5: Frontend Development
- Build all React components
- Implement responsive design
- Add data visualizations

### Phase 6: Testing & Refinement
- Unit tests, integration tests
- Bug fixes and optimization

### Phase 7: Documentation & Deployment
- Write documentation
- Prepare for deployment

---

## Important Implementation Notes

### ESPN API Integration
- Use the `espn-api` Python package (NOT web scraping ESPN directly)
- Initialize with: `League(league_id, year, espn_s2, swid)`
- Handles: league settings, rosters, free agents, matchups, player stats
- Implement retry logic and error handling
- Cache responses to minimize API calls

### Basketball Reference Scraping
- Use BeautifulSoup4 to parse HTML tables
- Respect robots.txt and rate limiting (1-2 requests/second)
- Parse per-game stats, advanced stats, shooting splits
- Store in database for quick access
- Handle special characters in player names

### Projection Models
- Pre-trained models saved as .pkl files in `backend/projections/trained_models/`
- Features: age, experience, minutes, usage rate, team pace, position
- Algorithms: Gradient Boosting / Random Forest for counting stats, Ridge/Lasso for shooting %
- **Tiered weighting system** based on games played (see Section 4 above)
- Weights are fixed by design - no manual intervention needed during season
- Annual retraining before each NBA season

### Injury & Game Rate Handling
- **Injury Source:** ESPN's `kona_playercard` endpoint (via espn-api library)
- **No Phantom Injuries:** Only ESPN-reported injuries affect projections
- **Schedule-Based:** Remaining games calculated from actual NBA team schedules
- **Adaptive Mode:** 0-4 GP = 90% rate (grace period), 5+ GP = actual rate with 75% floor
- **Flat Rate Mode:** User-configurable fixed percentage for all players
- **Projected Games:** Enforced in start limit optimizer to prevent over-projection

### Security
- Password hashing with bcrypt (12+ rounds)
- Encrypt ESPN cookies at rest
- Parameterized SQL queries (SQLAlchemy ORM handles this)
- CSRF protection on forms
- Rate limiting on API endpoints

### Performance
- Database indexing on league_id, player_id, team_id
- Connection pooling
- Cache ESPN API responses (Redis or in-memory)
- Async tasks for long projections

---

## Code Style & Best Practices

### Python (Backend)
- Follow PEP 8 style guide
- Use type hints where appropriate
- Docstrings for all functions/classes
- Meaningful variable names
- Error handling with try/except and logging
- Use environment variables for secrets (python-dotenv)

### JavaScript/React (Frontend)
- Functional components with hooks
- PropTypes or TypeScript for type checking
- Component file naming: PascalCase (Dashboard.js)
- Use async/await for API calls
- Handle loading and error states in UI

### Git
- Commit frequently with descriptive messages
- Branch for features: `feature/trade-analyzer`, `feature/projections`
- .gitignore: node_modules/, venv/, .env, *.pyc, __pycache__/

---

## Environment Variables (.env)

```bash
# Flask
FLASK_APP=backend/app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=sqlite:///fantasy_basketball.db  # Development
# DATABASE_URL=postgresql://user:pass@localhost/fantasy_basketball  # Production

# Anthropic API (if needed for advanced features)
ANTHROPIC_API_KEY=your-api-key

# Scheduler
REFRESH_SCHEDULE_TIME=03:00:00
```

---

## Testing Guidelines

### Backend Tests
```bash
# Run tests
python -m pytest backend/tests/

# With coverage
python -m pytest --cov=backend backend/tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Manual Testing
- Test with real ESPN leagues (both H2H and Roto)
- Test edge cases: empty leagues, mid-season, injuries
- Browser testing: Chrome, Firefox, Safari
- Mobile responsive testing

---

## Common Commands

### Backend
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask app
flask run
```

### Database Migrations (Flask-Migrate)
```bash
# Initialize migrations (first time only)
flask db init

# Create a new migration after model changes
flask db migrate -m "Add projection_method to leagues"

# Apply migrations to database
flask db upgrade

# Downgrade to previous migration
flask db downgrade

# Show current migration version
flask db current

# Show migration history
flask db history
```

**Migration Setup (already configured):**
- Flask-Migrate is installed and configured in `app.py`
- Migrations stored in `migrations/` directory
- `Migrate(app, db)` initialization in app factory

### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Run dev server
npm start

# Build for production
npm run build
```

---

## Debugging Tips

### ESPN API Issues
- Verify ESPN_S2 and SWID cookies are correct and not expired
- Check league_id and year parameters
- Enable debug logging to see API requests/responses
- Test with a simple league first

### Database Issues
- Check SQLAlchemy connection string
- Verify tables exist: `sqlite3 fantasy_basketball.db ".tables"`
- Check migrations are up to date
- Look at query logs for slow queries

### Projection Issues
- Verify training data quality
- Check for missing/null values
- Validate feature engineering
- Compare predictions to actual outcomes
- Check model file paths and loading

---

## Resources & References

- **ESPN API Package:** https://github.com/cwendt94/espn-api
- **Basketball Reference:** https://www.basketball-reference.com/
- **Flask Docs:** https://flask.palletsprojects.com/
- **React Docs:** https://react.dev/
- **SQLAlchemy Docs:** https://docs.sqlalchemy.org/
- **scikit-learn Docs:** https://scikit-learn.org/

---

## User Information

**Developer:** Milo
- Recent St. Lawrence University graduate (BS Mathematics, minors CS & European Studies)
- Currently pursuing Master's in Data Science and Statistics
- Technical background: R, SQL, Python
- Experience: Salesforce data management (Clinch internship)
- Interest: Baseball analytics, sports statistics
- Rocket League esports team captain
- Completed capstone: K-Nearest Neighbors for MLB WAR prediction

---

## Current Status

**Phase:** Phase 4 - Core Features (Complete)
**Completed:**
- Project structure and Flask backend
- SQLite database with migrations (Flask-Migrate)
- ESPN integration with espn-api
- Injury handling from ESPN kona_playercard endpoint
- Projection method settings (adaptive/flat rate)
- Schedule-based remaining games calculation
- Start limit optimizer with projected_games enforcement
- Z-score value system implementation
- Trade analyzer with category impact analysis
- Trade suggestions generator with configurable aggressiveness
- Multi-player trade support (2-for-1, 3-for-2, etc.)
- Automatic roster management with smart drops
- Dynamic roster limit fetching from ESPN
- Dashboard with standings, projections, quick insights
- Trade Opportunities section with inline settings
- Category analysis (strengths/weaknesses)
- Stats/Points view toggle in standings tables
- Auto-refresh on settings change
- Tie handling for Roto category rankings
- Counting stats rounding (STL, BLK, etc.)
- FG%/FT% z-score fixes in trade analyzer
- Waiver Wire Analyzer with click-to-analyze
- Availability filtering (filters out injured/suspended players)

**Current State:**
- Trade analyzer fully functional with multi-player support
- Waiver wire analyzer complete with z-score based add/drop
- All major z-score inconsistencies resolved
- Dashboard UI polished with responsive settings
- Roto standings calculations are accurate with proper tie handling
- Dynamic roster limits working (excludes IR slots)

**Next Steps:**
1. **Player Rankings (Next Priority)** - Universal z-score based player rankings
2. H2H matchup analysis improvements
3. Historical Performance Tracking
4. Mobile responsive design refinements
5. ML model training and integration
6. Email notifications (future)

---

## Important Reminders for Claude Code

1. **Reference the PRD:** The full PRD file (fantasy-basketball-optimizer-PRD.md) has complete specifications for every feature
2. **Ask questions:** If unclear about implementation, ask Milo before proceeding
3. **Commit frequently:** Git commit after each major component
4. **Test as you go:** Write tests alongside features
5. **Document:** Add comments and docstrings
6. **Security first:** Never commit .env files, encrypt sensitive data
7. **Think modular:** Build reusable components and functions

---

**Last Updated:** March 14, 2026
**Version:** 1.7 (Added Player Rankings documentation as next priority)

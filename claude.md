# Claude Code Context - Fantasy Basketball Optimizer

## Project Overview
A **Fantasy Basketball Optimizer** web application that integrates with ESPN Fantasy Basketball to provide advanced analytics, projections, and recommendations.

**Key Goal:** Parse ESPN league data to project end-of-season standings, recommend trades, and suggest waiver wire acquisitions using z-score based analysis.

---

## Critical Files
- **fantasy-basketball-optimizer-PRD.md** - Complete Project Requirements Document
- **This file (CLAUDE.md)** - Quick reference and context

---

## Tech Stack

### Backend
- **Python 3.10+** with **Flask** web framework
- **SQLAlchemy** ORM with **Flask-Migrate** (Alembic)
- **SQLite** (development) / **PostgreSQL** (production)
- **espn-api** - ESPN Fantasy Basketball API wrapper
- **pandas/numpy** - Data analysis
- **Flask-Login** - Authentication

### Frontend
- **React 18+** with **React Router**
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **CSS** - Custom styling

### Database
- **SQLite** for local development
- Migrations managed via Flask-Migrate

---

## Current Status (October 2026)

### Completed Features
- ESPN integration with espn-api package
- Injury handling from ESPN kona_playercard endpoint
- Projection method settings (adaptive/flat rate)
- Schedule-based remaining games calculation
- Start limit optimizer with z-score player values
- **Player Rankings** - All NBA players ranked by z-score (complete)
- **Trade Analyzer** - Multi-player trades with roster management (complete)
- **Waiver Wire Analyzer** - Z-score based add/drop analysis (complete)
- **ESPN-style Contribution Z-Scores** - Volume-aware FG%/FT% (complete)
- **Keepers** - Keeper league rankings with age, z-scores, team selector (complete)
- **Automatic Season Transitions** - Dynamic date handling with NBASeasonDates utility (complete)
- Dashboard with standings, projections, quick insights
- Roto tie handling, stats view toggle, auto-refresh

### Architecture Decisions
- Z-scores replace hardcoded category weights throughout
- ESPN-style contribution method for FG%/FT% calculations
- In-memory caching (24-hour) for Player Rankings
- Dynamic roster limits from ESPN (excludes IR slots)
- Centralized NBASeasonDates utility for all date calculations
- Automatic season detection and transition on dashboard load

---

## Z-Score Methodology

### ESPN-Style Contribution Z-Scores (Current Implementation)

FG% and FT% use **contribution-based z-scores** that account for both volume and efficiency:

```python
# Calculate league baseline percentage
league_avg_fg_pct = total_fgm / total_fga  # e.g., 0.462

# Player contribution = makes above/below expected
expected_fgm = player_fga * league_avg_fg_pct
contribution = player_fgm - expected_fgm

# Z-score from contribution
z_score = (contribution - mean_contribution) / std_contribution
```

**Why This Matters:**
- High volume + good efficiency = big positive z-score (rewarded)
- High volume + bad efficiency = big negative z-score (penalized)
- Low volume = muted impact either way
- Matches ESPN Player Rater and Basketball Monster methodology

### Z-Score Contexts

| Context | Data Source | Purpose |
|---------|-------------|---------|
| **Player Rankings** | Historical per-game stats vs ALL NBA players | Reference: who's been best this season |
| **Start Limit Optimizer** | ROS projections vs rostered players | Optimize lineup decisions |
| **Trade/Waiver Analysis** | Optimizer's ROS z-scores | Future-oriented decisions |

**Key Insight:** Player Rankings uses historical data (all NBA players, 1+ game minimum), while Optimizer uses forward-looking ROS projections (rostered players only).

---

## Key Features

### 1. Player Rankings (Complete)

Universal player rankings for all NBA players using league-specific z-scores.

**Features:**
- All players with ≥1 game played (~400+ players)
- Sortable by total z-score or any category
- Position filtering (PG, SG, SF, PF, C)
- Search by player name or team
- "FA" badge for free agents
- Injury status indicators (DTD, O, IR)
- 24-hour in-memory cache per league

**API Endpoint:** `GET /api/leagues/{id}/player-rankings`

**Key Files:**
- `backend/api/players.py` - Rankings endpoint and z-score calculation
- `frontend/src/components/PlayerRankings.js` - UI component

### 2. Multi-Player Trade Support (Complete)

Analyze trades of any size with automatic roster management.

**Supported Trade Types:**
- 1-for-1, 2-for-1, 2-for-2, 3-for-1, 3-for-2, etc.

**Automatic Roster Management:**
When receiving more players than giving:
1. Calculate overflow: `players_received - players_given`
2. Identify drop candidates (exclude traded players)
3. Sort by z-score (lowest first)
4. Auto-recommend dropping N lowest z-score players

**Dynamic Roster Limit:**
- Fetched from ESPN league settings
- Excludes IR slots (IR, IR+, IL, IL+)
- Cached in `League.active_roster_limit`

**API Endpoint:** `POST /api/leagues/{id}/trades/analyze`

**Key Files:**
- `backend/analysis/trade_analyzer.py` - `_calculate_additional_drops()`
- `backend/api/trades.py` - Roster limit handling
- `frontend/src/components/QuickInsights.js` - TradeAnalyzerModal

### 3. Waiver Wire Analyzer (Complete)

Z-score based add/drop evaluation with availability filtering.

**Features:**
- Click any waiver target for full analysis
- Compares free agent z-score to roster's worst player
- Net benefit: `free_agent_z - drop_candidate_z`
- Category impact analysis

**Availability Filtering:**
Excludes unavailable players:
- Out for season
- Expected return > 14 days
- Suspended or inactive status

**Key Files:**
- `backend/analysis/waiver_analyzer.py` - WaiverAnalyzer class
- `backend/api/waivers.py` - API endpoints with filtering

### 4. Start Limit Optimizer (Roto)

Season simulation for leagues with position start limits using player-first optimization.

**Player-First Algorithm:**
1. Sort all players by z-score (highest first)
2. For each player, assign to slots for ALL their remaining games
3. Use most-specific-position-first slot assignment (PG → G → UTIL)
4. Skip if no eligible slots or player reached projected games limit

**Why Player-First?**
- Guarantees top players (highest z-score) start every game they can
- Better results than day-by-day greedy approach
- Preserves roster flexibility for lower-priority players

**Injury Handling:**
- Only "OUT" status excludes players (not DTD/Questionable)
- OUT with return date: excluded until that date
- OUT without return date: assumed return tomorrow
- Real-time data from ESPN `kona_playercard` view

**Key Files:**
- `backend/projections/start_limit_optimizer.py`
- `simulate_season_player_first()` - Main optimization algorithm
- `calculate_league_averages()` - League baseline calculation
- `calculate_player_value()` - Z-score value computation

### 5. Daily Lineup Manager (Complete)

Day-by-day lineup visualization showing optimized player assignments.

**Features:**
- Position-by-position lineup for any date
- Date navigation (Previous/Today/Next)
- Sections: Starting Lineup, Bench, Injured (OUT), No Game, IR
- Only for Roto leagues with start limits enabled

**Injured Section Logic:**
- Shows OUT players whose teams have games today
- Checks NBA schedule to verify team actually plays
- DTD/Questionable players NOT shown (expected to play)
- Orange styling for visual distinction

**Dashboard Tab Navigation:**
- Four tabs: Overview | Daily Lineup | Player Rankings | Keepers (if keeper league)
- Lazy loading: components render only when tab active
- Daily Lineup tab only appears for Roto leagues with start limits
- Keepers tab only appears for keeper leagues

**API Endpoint:** `GET /api/leagues/{id}/daily-lineup?date=YYYY-MM-DD`

**Key Files:**
- `backend/api/daily_lineup.py` - Endpoint with 5-minute caching
- `frontend/src/components/DailyLineup.js` - UI component
- `frontend/src/components/LeagueDashboard.js` - Tab navigation

### 6. Keepers (Complete)

Keeper league rankings to help users evaluate roster players for keeper selection.

**Features:**
- Auto-detects keeper leagues from ESPN settings
- Shows keeper count allowed per league
- Team selector dropdown to view any team's roster
- Ranks roster players by total z-score
- Shows player age (fetched from ESPN athlete API)
- Shows games played, positions, injury status
- Per-category z-scores for all league categories
- Sortable by any column including categories
- Age color-coding: green (≤25), red (≥32)
- Highlights recommended keepers (top N by z-score)

**Age Fetching:**
- Ages fetched from ESPN public athlete API
- Cached per-season in `player_birth_dates` table
- Cache invalidates on new season

**API Endpoints:**
- `GET /api/leagues/{id}/keepers?team_id=X` - Get keeper rankings
- `GET /api/leagues/{id}/keeper-settings` - Check if keeper league

**Key Files:**
- `backend/api/keepers.py` - Keepers endpoint
- `backend/services/espn_client.py` - `get_player_ages()`, `fetch_player_age()`
- `backend/models.py` - `PlayerBirthDate` model
- `frontend/src/components/Keepers.js` - UI component

---

## Database Schema (Key Tables)

### Leagues
```
- id, user_id, espn_league_id, season, league_name
- espn_s2_cookie, swid_cookie (encrypted)
- league_type (H2H_CATEGORY, H2H_POINTS, ROTO)
- roster_settings (JSONB), scoring_settings (JSONB)
- projection_method ("adaptive" or "flat_rate")
- flat_rate_value (decimal, 0.0-1.0)
- trade_suggestion_mode ("conservative", "normal", "aggressive")
- active_roster_limit (integer, cached from ESPN)
```

---

## Technical Notes

### NBASeasonDates Utility

Centralized utility for all season date calculations, eliminating hardcoded dates across the codebase.

**File:** `backend/utils/season_dates.py`

```python
from backend.utils.season_dates import NBASeasonDates

# Get current season (Oct-Dec = year+1, Jan-Sep = year)
season = NBASeasonDates.get_season_from_date()  # e.g., 2027 for Oct 2026

# Get season boundaries
start = NBASeasonDates.get_season_start(2027)  # date(2026, 10, 22)
end = NBASeasonDates.get_season_end(2027)      # date(2027, 4, 13)

# Check season status
NBASeasonDates.is_season_complete(2027)  # True if past end date
NBASeasonDates.get_season_progress(2027) # 0.0 to 1.0

# All-Star break (3rd weekend of February)
start, end = NBASeasonDates.get_all_star_break(2027)
```

**Season Identification:**
- NBA seasons span two calendar years (e.g., 2026-27 season)
- ESPN uses the ending year as identifier (2027 for 2026-27 season)
- October-December dates → next year's season
- January-September dates → current year's season

**Files Using NBASeasonDates:**
- `backend/api/daily_lineup.py` - Current season detection
- `backend/api/dashboard.py` - Games remaining estimation
- `backend/api/leagues.py` - New season detection
- `backend/api/season_recap.py` - Season completion check
- `backend/projections/hybrid_engine.py` - Season progress
- `backend/projections/simple_projection.py` - Season dates
- `backend/projections/start_limit_optimizer.py` - Season boundaries
- `backend/scrapers/nba_schedule.py` - Schedule fetching

### FG%/FT% Z-Score Formula
```python
# League baseline
league_avg_pct = sum(all_makes) / sum(all_attempts)

# Player contribution
contribution = makes - (attempts * league_avg_pct)

# Z-score
z_score = (contribution - mean_contribution) / std_contribution
```

### Caching Strategy
- **Player Rankings:** In-memory dict, 24-hour TTL, key: `rankings_{league_id}`
- **ESPN API responses:** Cached where possible to minimize calls

### Roster Limit Handling
```python
# Excludes IR slots
IR_SLOT_NAMES = ['IR', 'IR+', 'IL', 'IL+']
active_roster_limit = total_slots - ir_slots
```

### Minimum Games Filter
- Player Rankings: 1 game minimum (include all active players)
- ROS Projections: Players with 0 projected games excluded

### Hybrid Projection Engine

Tiered weighting system for ROS projections based on games played:

| Tier | Games Played | ESPN Projection | Current Stats |
|------|-------------|-----------------|---------------|
| 1    | 0-5         | 100%            | 0%            |
| 2    | 6-15        | 60%             | 40%           |
| 3    | 16-35       | 15%             | 85%           |
| 4    | 36+         | 0%              | 100%          |

**Rationale:**
- Early season: Trust ESPN projections (limited sample size)
- Mid-season: Blend both sources
- Late season: Trust actual performance data

**Key File:** `backend/projections/hybrid_engine.py`

---

## API Endpoints

### Player Rankings
- `GET /api/leagues/{id}/player-rankings` - Get all ranked players
- `POST /api/leagues/{id}/player-rankings/clear-cache` - Clear cache

### Trades
- `POST /api/leagues/{id}/trades/analyze` - Analyze trade
- `GET /api/leagues/{id}/trades/suggestions` - Get suggestions

### Waivers
- `POST /api/leagues/{id}/waivers/analyze` - Analyze add/drop
- `GET /api/leagues/{id}/waivers/suggestions` - Get suggestions

### Keepers
- `GET /api/leagues/{id}/keepers?team_id=X` - Get keeper rankings for team
- `GET /api/leagues/{id}/keeper-settings` - Check keeper league status

### Dashboard
- `GET /api/leagues/{id}/dashboard` - Full dashboard data

---

## Common Commands

### Backend
```bash
source venv/bin/activate
pip install -r requirements.txt
flask run --port 5001
```

### Frontend
```bash
cd frontend
npm install
npm start  # Runs on port 3000
```

### Database Migrations
```bash
flask db migrate -m "Description"
flask db upgrade
```

---

## Bug Fixes Log

### October 2026
- **Week 105 bug:** Fixed season not transitioning automatically
- **Hardcoded dates:** Replaced with centralized NBASeasonDates utility
- **ML code removal:** Simplified hybrid engine to ESPN + current stats only

### March 2026
- **Auto-drop logic:** Fixed 0 vs None handling for z-scores
- **Roster limits:** Dynamic fetching from ESPN (excludes IR)
- **Contribution z-scores:** Implemented ESPN-style FG%/FT% calculation

### February 2026
- **Roto tie handling:** Average ranks for tied teams
- **Counting stats:** Rounded to whole numbers
- **FG%/FT% z-scores:** Component-based calculation fix

---

## Next Steps

1. H2H matchup analysis improvements
2. Historical performance tracking
3. Mobile responsive refinements
4. Email notifications (future)

---

**Last Updated:** October 23, 2026
**Version:** 2.2 (Automatic season transitions with NBASeasonDates utility)

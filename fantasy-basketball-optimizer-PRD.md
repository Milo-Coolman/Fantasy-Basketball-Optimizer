# Fantasy Basketball Optimizer - Project Requirements Document

## 1. Project Overview

### 1.1 Project Name
Fantasy Basketball Optimizer

### 1.2 Project Description
A web application that integrates with ESPN Fantasy Basketball to provide advanced analytics, projections, and recommendations. The app analyzes league data to project end-of-season standings, recommend trades, and suggest waiver wire acquisitions.

### 1.3 Target Users
- ESPN Fantasy Basketball league participants
- Users seeking data-driven insights for roster management
- Both Head-to-Head (H2H) and Rotisserie (Roto) league formats

### 1.4 Core Value Proposition
Leverage machine learning and statistical analysis to provide actionable insights that improve fantasy basketball team performance through:
- Accurate season-ending projections
- Data-driven trade recommendations
- Optimized waiver wire pickup suggestions

---

## 2. Technical Architecture

### 2.1 Technology Stack

**Backend:**
- Python 3.10+
- Flask (web framework)
- SQLAlchemy (ORM)
- Flask-Migrate (database migrations via Alembic)
- PostgreSQL (production) / SQLite (development)
- espn-api package (ESPN Fantasy Basketball API wrapper)
- scikit-learn (machine learning)
- pandas/numpy (data analysis)
- BeautifulSoup4 + requests (web scraping)
- Flask-Login (authentication)
- APScheduler (scheduled tasks)

**Frontend:**
- React 18+
- React Router (navigation)
- Axios (HTTP client)
- Chart.js or Recharts (data visualization)
- Tailwind CSS or Material-UI (styling)
- React Context API or Redux (state management)

**Database:**
- SQLite (local development)
- PostgreSQL (future production deployment)

**Deployment:**
- Local development environment initially
- Future: Heroku, AWS, or Vercel

### 2.2 Project Structure
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
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── trade_analyzer.py       # Trade impact analysis
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
│   │   │   ├── Dashboard.js
│   │   │   ├── StandingsTable.js
│   │   │   ├── ProjectionsChart.js
│   │   │   ├── TradeAnalyzer.js
│   │   │   ├── WaiverRecommendations.js
│   │   │   ├── WeeklyMatchups.js   # H2H only
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
└── README.md
```

---

## 3. Feature Requirements

### 3.1 User Authentication & Management

**3.1.1 User Registration**
- Users create account with email and password
- Password hashing and secure storage
- Email validation
- Success/error messaging

**3.1.2 User Login**
- Email and password authentication
- Session management with Flask-Login
- "Remember me" functionality
- Password reset capability (future enhancement)

**3.1.3 ESPN Account Integration**
- User provides ESPN league URL
- User provides ESPN_S2 and SWID cookies
- Clear documentation with screenshots for cookie extraction
- Validation of ESPN credentials
- Support for multiple leagues per user

### 3.2 League Data Management

**3.2.1 League Setup**
- Parse ESPN league URL to extract league ID and season
- Fetch and store league settings:
  - League name
  - Number of teams
  - Roster positions (PG, SG, SF, PF, C, G, F, UTIL, BE)
  - Scoring format (H2H Categories, H2H Points, Roto)
  - Scoring categories/points system
  - Playoff structure (teams, weeks)
  - Trade deadline
  - Waiver system (FAAB, rolling waivers, etc.)

**3.2.2 Data Retrieval**
- Fetch all team rosters with current players
- Retrieve available free agents
- Get current standings
- Pull historical matchup results (H2H leagues)
- Cache data to reduce API calls
- Daily automatic refresh at configurable time (default: 3 AM local)
- Manual refresh button for immediate updates

**3.2.3 Data Storage**
- Store league settings in database
- Cache player statistics and projections
- Track historical data for model training
- Store user's league associations

### 3.3 Projection Engine

**3.3.1 Data Sources**
- ESPN Fantasy API (current stats, ESPN projections)
- Basketball Reference (advanced stats, historical data)
- Cached historical fantasy data for ML training

**3.3.2 Statistical Projection Component**
- Aggregate rest-of-season projections from multiple sources
- Weight projections by source reliability
- Adjust for:
  - Games remaining in season
  - Recent performance trends (last 15 games)
  - Injury status
  - Team schedule strength

**3.3.3 Machine Learning Component**
- Pre-trained models using historical NBA and fantasy data
- Features include:
  - Player age, experience
  - Usage rate, minutes per game
  - Team pace, offensive/defensive ratings
  - Historical consistency metrics
  - Position scarcity factors
- Model types:
  - Regression models for counting stats (points, rebounds, assists, etc.)
  - Classification for categorical outcomes (win probability)

**3.3.4 Hybrid Projection Engine**

The hybrid engine combines three data sources using a **tiered weighting system** based on games played. This approach recognizes that early-season projections should rely on preseason expectations, while late-season projections should primarily use actual current performance.

**Data Sources:**
1. **ESPN Projections** - Preseason and in-season projections from ESPN Fantasy
2. **Current Season Stats** - Actual performance this season
3. **ML Model** - Machine learning predictions based on player features

**Note:** Previous season stats are not used because the ESPN API does not provide them when connected to the current season. The espn-api library only returns stats for the season you connect to.

**Tiered Weighting Table:**

| Tier | Games Played | ESPN Proj | Current Season | ML Model |
|------|--------------|-----------|----------------|----------|
| 1    | 0-5 games    | 90%       | 0%             | 10%      |
| 2    | 6-15 games   | 55%       | 35%            | 10%      |
| 3    | 16-35 games  | 15%       | 80%            | 5%       |
| 4    | 35+ games    | 0%        | 100%           | 0%       |

**Tier Logic Explained:**
- **Tier 1 (0-5 games):** Very small sample size. ESPN projections are the primary source (90%). Current stats are ignored due to unreliability. ML provides minor adjustment (10%).
- **Tier 2 (6-15 games):** Sample size becomes meaningful. Current season starts contributing (35%), balanced with ESPN projections (55%). ML provides context (10%).
- **Tier 3 (16-35 games):** Current performance becomes dominant (80%) as patterns emerge. ESPN projections (15%) provide stability. ML contribution reduced (5%).
- **Tier 4 (35+ games):** Large sample size provides reliable data. Current season performance used exclusively (100%). Other sources are no longer needed.

**Projection Output - Component Stats:**

The engine outputs 10 component statistics per player:
- **Counting Stats:** PTS, REB, AST, STL, BLK, 3PM
- **Shooting Components:** FGM, FGA, FTM, FTA

These 10 stats are used to calculate the 8 standard fantasy scoring categories:
- PTS, REB, AST, STL, BLK, 3PM (direct from projections)
- FG% (calculated as FGM ÷ FGA)
- FT% (calculated as FTM ÷ FTA)

**Percentage Stat Calculation:**

When combining projections from multiple sources, percentage stats are calculated from weighted component totals (not averaged):

```
Example: Combining ESPN projection + Current season for FG%

ESPN: 5.2 FGM / 11.0 FGA (47.3%)
Current: 4.8 FGM / 10.5 FGA (45.7%)

With 50/50 weighting:
- Combined FGM = (5.2 × 0.5) + (4.8 × 0.5) = 5.0
- Combined FGA = (11.0 × 0.5) + (10.5 × 0.5) = 10.75
- Combined FG% = 5.0 ÷ 10.75 = 46.5%

This is more accurate than averaging 47.3% and 45.7% directly.
```

**Additional Adjustments:**
- League-specific adjustments based on scoring settings
- Confidence intervals for projections
- Rest-of-season totals scaled by remaining games

**3.3.5 H2H League Projections**
- Week-by-week matchup projections
- Category-by-category win probability
- Playoff probability calculations
- Simulate remaining season 10,000 times (Monte Carlo)
- Account for schedule variance and opponent strength

**3.3.6 Roto League Projections**
- End-of-season category totals projection
- Category rank projections (1st-12th in each category)
- Overall league win probability
- Identify strengths/weaknesses by category

**3.3.7 Injury Detection and Handling**

The system uses ESPN's injury data to accurately project player availability and remaining games.

**Injury Data Source:**
- Injury information is fetched from ESPN's `kona_playercard` endpoint via the espn-api library
- This provides real-time injury status, expected return dates, and injury details
- Players are only marked as injured when ESPN reports an actual injury status

**Injury Status Detection:**
- **Active/Healthy:** No injury status from ESPN - player projected for full remaining games
- **Day-to-Day (DTD):** Minor injury, may miss games - projections adjusted accordingly
- **Out (O):** Currently injured and missing games - projections pause until return
- **Injured Reserve (IR):** Long-term injury, player in IR slot (`lineupSlotId == 13`)

**Projected Games Calculation:**
For each player, remaining games are calculated from:
1. **Team Schedule:** Actual NBA games remaining for the player's team (from schedule data)
2. **Game Rate:** Player's games played / team games played (capped at 100%)
3. **Projection Method:** Either adaptive or flat rate (see Section 3.3.9)
4. **Injury Adjustment:** If injured, remaining games = 0 until return date

**Key Implementation Notes:**
- Phantom injuries removed: Only ESPN-reported injuries affect projections
- Schedule-based: Uses actual team schedules, not estimated 82-game totals
- Return dates: ESPN provides expected return dates for injured players
- IR slot detection: `lineupSlotId == 13` identifies players on IR

**3.3.8 IR Player Return Projections**

For players on Injured Reserve with expected return dates:

**Return Projection Logic:**
1. Retrieves the expected return date from ESPN's injury timeline
2. Calculates team games remaining from that date forward
3. Projects the IR player will be activated on their return date
4. Runs drop decision analysis to determine optimal roster move

**IR Drop Optimizer (Z-Score Based):**
When an IR player is projected to return, the system recommends which active roster player to drop using a simplified z-score comparison:

- **Z-Score Value:** Each player has a per-game z-score value (see Section 3.3.10)
- **Drop Logic:** Drop the player with the **lowest z-score value** (worst performer)
- **Net Gain Calculation:** `gain = ir_player_z_value - drop_player_z_value`
- **Example Decision:**
  ```
  IR Player: Kawhi Leonard (z-value: +3.45/game)
  Drop Candidates:
    1. Buddy Hield: z-value=-0.82/game (gain: +4.27)
    2. Dillon Brooks: z-value=-0.45/game (gain: +3.90)
    3. Marcus Smart: z-value=+0.12/game (gain: +3.33)

  OPTIMAL DROP: Buddy Hield (lowest z-score, highest gain)
  ```
- **Execution Speed:** Instant results (vs. 10-30 seconds with full Roto simulation)

**Key Considerations:**
- Only analyzes IR returns for players with `projected_games > 0`
- Z-scores capture relative value across all categories automatically
- Respects ESPN's droppable flag for each player
- Handles multiple IR returns by excluding already-dropped players

**3.3.9 Projection Method Settings**

Users can configure how game rates are calculated for projecting remaining games. This setting affects the `projected_games` value used in all projections.

**Adaptive Mode (Default):**
Adjusts the game rate calculation based on how many games a player has played:

| Games Played | Game Rate Calculation |
|--------------|----------------------|
| 0-4 games    | 90% of remaining team games (grace period) |
| 5+ games     | Actual rate (GP ÷ Team GP) with 75% floor |

**Grace Period (0-4 games):** New players, recently traded players, or players returning from early-season injuries haven't had enough games to establish a reliable rate. The 90% assumption provides a reasonable starting point.

**75% Floor:** Players with 5+ games use their actual game rate, but never below 75%. This prevents over-penalizing players who missed games due to minor injuries or rest days early in the season.

**Example:**
```
Player A: 40 GP out of 50 team games = 80% rate → Uses 80%
Player B: 30 GP out of 50 team games = 60% rate → Uses 75% (floor applied)
Player C: 3 GP (new to team) → Uses 90% (grace period)
```

**Flat Rate Mode:**
Applies a user-specified percentage to all players uniformly:
- User sets a fixed game rate (e.g., 85%)
- All healthy players are projected for `remaining_team_games × flat_rate`
- Ignores individual player game rates
- Useful for simplified projections or testing scenarios

**Settings Storage:**
- `projection_method`: "adaptive" or "flat_rate"
- `flat_rate_value`: Decimal (0.0-1.0) used when method is "flat_rate"
- Stored per league in the `leagues` table

**3.3.10 Day-by-Day Start Limit Optimization**

Roto leagues typically enforce position start limits (commonly 82 games per position across the full NBA season). The day-by-day optimizer ensures projections accurately reflect what players will actually contribute given these constraints.

**Start Limit Concept:**
- Each roster position (PG, SG, SF, PF, C, G, F, UTIL) has a maximum number of starts allowed
- Full-season leagues typically allow 82 starts per position (one per NBA game day)
- Players exceeding the limit for their position cannot contribute additional stats

**Projected Games Enforcement:**
The optimizer respects each player's `projected_games` limit:
- A player can only be assigned starts up to their `projected_games` value
- This prevents over-projecting players expected to miss games due to rest or injury history
- `projected_games` is calculated using the projection method settings (adaptive or flat rate)

**3.3.11 Z-Score Based Player Value System**

The optimizer uses z-scores to calculate each player's per-game fantasy value, enabling fair comparison across all categories regardless of scale.

**League-Wide Averages Calculation:**
Before simulating the season, the optimizer calculates league averages from all rostered players:
1. Collects per-game stats for all players across all teams in the league
2. Calculates mean and standard deviation for each scoring category
3. Caches these values for consistent z-score calculations

**Z-Score Formula:**
```
z_score = (player_stat - league_mean) / league_std_dev
```

For each player, per-game value is calculated as:
```
per_game_value = Σ z_scores across all categories
```

**Category Handling:**
- **Counting Stats (PTS, REB, AST, STL, BLK, 3PM):** Higher values = positive z-score
- **Turnovers (TO):** Sign is flipped (lower TO = positive z-score)
- **Percentage Stats (FG%, FT%):** Multiplied by 100 before z-score calculation
  - Converts 0.476 → 47.6 to match scale of counting stats
  - Ensures fair weighting between percentages and counting stats

**Example Z-Score Calculation:**
```
League Averages: PTS=15.2 (std=6.5), REB=5.8 (std=2.3), AST=3.4 (std=2.1)

LeBron James: 25.5 PTS, 7.2 REB, 8.1 AST
  PTS z-score: (25.5 - 15.2) / 6.5 = +1.58
  REB z-score: (7.2 - 5.8) / 2.3 = +0.61
  AST z-score: (8.1 - 3.4) / 2.1 = +2.24
  ... (repeat for all 8-9 categories)
  Total z-value: +8.5/game (elite player)

Buddy Hield: 12.8 PTS, 3.2 REB, 2.1 AST
  PTS z-score: (12.8 - 15.2) / 6.5 = -0.37
  REB z-score: (3.2 - 5.8) / 2.3 = -1.13
  AST z-score: (2.1 - 3.4) / 2.1 = -0.62
  ... (repeat for all categories)
  Total z-value: -0.82/game (below average)
```

**Why Z-Scores?**
- **League-Specific:** Averages are calculated from YOUR league's roster players
- **Category Scarcity:** Naturally weights scarce categories higher (smaller std = bigger z-score impact)
- **Fair Comparison:** 25 PTS and 2.5 BLK can be compared on the same scale
- **No Manual Tuning:** Replaces arbitrary hardcoded category weights

**Day-by-Day Simulation Approach:**

For each remaining day in the NBA season, the optimizer:

1. **Check Schedule:** Identifies which rostered players have NBA games that day
2. **Check Player Limits:** Verifies player hasn't exceeded their `projected_games` limit
3. **Assign Starters:** For each roster slot, assigns the highest z-value eligible player:
   - Player must have a game that day
   - Player must be eligible for that position
   - Player must not have exceeded their projected games
   - Position must not have reached its start limit
4. **Track Usage:** Increments `starts_used` counter for the assigned position and player
5. **Bench Players:** Players not assigned to a starting slot are "benched" and contribute 0 stats for that day
6. **Enforce Limits:** Once a position or player reaches their limit, no more starts assigned

**Optimization Logic:**
- **Position Eligibility:** Players can only fill slots they're eligible for (e.g., a PG/SG can fill PG, SG, G, or UTIL)
- **Z-Score Priority:** Higher z-value players get assigned first to maximize total value
- **Conflict Resolution:** When multiple players compete for the same slot, assigns based on z-score per-game value
- **Schedule Awareness:** Only considers players whose NBA teams play on each specific day

**Projection Output:**
- Provides realistic expected stats based on actual games a player will start
- Accounts for position conflicts (e.g., too many guards competing for limited G/UTIL slots)
- Prevents over-projection by not assuming players play all their remaining games
- Shows position-by-position start allocation and remaining starts

**Example Scenario:**
```
Position: PG (82 start limit)
Current date: February 1 (45 games remaining in season)
Starts used: 50
Starts remaining: 32

Roster has 2 PGs:
- Player A: 40 remaining games, z-value=+3.2/game
- Player B: 38 remaining games, z-value=+1.8/game

Optimizer assigns Player A to PG slot when both have games (higher z-value).
When only 32 more PG starts are available, some games will have
both players benched or assigned to alternate positions (G/UTIL).
```

**Benefits:**
- Accurate projections that respect roster constraints
- Z-score based prioritization ensures optimal player assignment
- League-specific value calculation adapts to your league's scoring landscape
- Identifies position scarcity issues (too many players, not enough starts)
- Helps users optimize roster construction for maximum stat accumulation
- Prevents misleading projections based on unrealistic playing time assumptions

### 3.4 Dashboard

**3.4.1 Overview Section**
- Current league standings
- Projected end-of-season standings
- User's team highlighted
- Playoff probability (H2H) or win probability (Roto)
- Last updated timestamp
- Manual refresh button

**3.4.2 Projections Visualization**
- **H2H Leagues:**
  - Week-by-week win probability chart
  - Playoff bracket projection
  - Upcoming matchup analysis
- **Roto Leagues:**
  - Category rankings bar chart (current vs projected)
  - Points breakdown by category
  - Gap to next rank in each category

**3.4.3 Team Insights**
- Roster strength by position
- Category performance breakdown
- Suggested focus areas for improvement

### 3.5 Trade Analyzer

**3.5.1 Trade Input Interface**
- Select user's team and trade partner
- Multi-player selection (any size trade, no limit)
- Support for 2-team trades only (no multi-team packages)
- Visual trade builder with drag-and-drop

**3.5.2 Trade Impact Analysis**
- **Pre-trade vs Post-trade comparison:**
  - Category value changes for both teams
  - Win probability impact (H2H) or rank impact (Roto)
  - Roster balance changes
- **Fairness assessment:**
  - Value differential calculation
  - Win-win vs lopsided trade indicator
- **League impact:**
  - How trade affects league standings
  - Playoff implications

**3.5.3 Trade Recommendations**
- AI-generated trade suggestions based on:
  - Team needs (weak categories)
  - Trade partner surpluses (strong categories)
  - Overall value optimization
- Ranked list of trade targets
- 1-for-1, 2-for-2, 3-for-3 suggestions
- "Trade value" score for each player

**3.5.4 Trade History**
- Log of analyzed trades
- Track suggested vs accepted trades
- Use data to improve recommendations over time

### 3.6 Waiver Wire Recommendations

**3.6.1 Available Players Analysis**
- Fetch all free agents from ESPN
- Calculate projected value for remainder of season
- Compare to current roster players

**3.6.2 Recommendation Algorithm**
- **Factors considered:**
  - Projected stats vs current roster
  - Streaming value (short-term adds)
  - Playoff schedule (games during fantasy playoffs)
  - Position scarcity
  - Injury replacements
  - Upside potential (breakout candidates)
- **Ranking system:**
  - Impact score (0-100)
  - Category fit for user's team
  - Confidence level

**3.6.3 Waiver Wire Interface**
- Ranked list of top available players
- Filter by position
- Sort by different metrics (projected points, impact score, etc.)
- "Add/Drop" suggestions showing best drop candidates
- Detailed player cards with:
  - Recent stats (last 7, 15, 30 days)
  - Rest-of-season projections
  - Injury status
  - Schedule analysis

**3.6.4 Streaming Recommendations (H2H)**
- Identify players with favorable weekly schedules
- Back-to-back and 4-game week flags
- Optimal streaming spots based on roster construction

### 3.7 Additional Features

**3.7.1 Matchup Analysis (H2H only)**
- Current week matchup breakdown
- Category-by-category projections
- Punt strategy suggestions (which categories to concede)
- Optimal lineup recommendations
- Streaming opportunities for the week

**3.7.2 Notifications & Alerts**
- Daily email digest with key insights (future)
- Trade deadline reminders
- Waiver wire targets becoming available
- Injury alerts affecting roster

**3.7.3 Settings & Preferences**
- Configure refresh schedule
- Email notification preferences
- Display preferences (dark mode, compact view, etc.)
- Multiple league management

---

## 4. Database Schema

### 4.1 Users Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
```

### 4.2 Leagues Table
```sql
CREATE TABLE leagues (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    espn_league_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    league_name VARCHAR(255),
    espn_s2_cookie TEXT NOT NULL,
    swid_cookie VARCHAR(255) NOT NULL,
    league_type VARCHAR(50), -- 'H2H_CATEGORY', 'H2H_POINTS', 'ROTO'
    num_teams INTEGER,
    roster_settings JSONB, -- Store roster positions as JSON
    scoring_settings JSONB, -- Store scoring rules as JSON
    playoff_settings JSONB, -- Playoff structure
    last_updated TIMESTAMP,
    refresh_schedule TIME DEFAULT '03:00:00',
    projection_method VARCHAR(20) DEFAULT 'adaptive', -- 'adaptive' or 'flat_rate'
    flat_rate_value DECIMAL(3,2) DEFAULT 0.85, -- Used when projection_method is 'flat_rate'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, espn_league_id, season)
);
```

### 4.3 Teams Table
```sql
CREATE TABLE teams (
    id SERIAL PRIMARY KEY,
    league_id INTEGER REFERENCES leagues(id) ON DELETE CASCADE,
    espn_team_id INTEGER NOT NULL,
    team_name VARCHAR(255),
    owner_name VARCHAR(255),
    current_record VARCHAR(50), -- "10-5-2" for H2H
    current_standing INTEGER,
    projected_standing INTEGER,
    win_probability DECIMAL(5,2), -- 0-100
    last_updated TIMESTAMP,
    UNIQUE(league_id, espn_team_id)
);
```

### 4.4 Players Table
```sql
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    espn_player_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    team VARCHAR(10), -- NBA team abbreviation
    position VARCHAR(20),
    injury_status VARCHAR(50),
    last_updated TIMESTAMP
);
```

### 4.5 Rosters Table
```sql
CREATE TABLE rosters (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
    player_id INTEGER REFERENCES players(id),
    acquisition_type VARCHAR(50), -- 'DRAFT', 'TRADE', 'WAIVER', 'FA'
    acquisition_date DATE,
    roster_slot VARCHAR(20), -- 'PG', 'SG', 'BE', etc.
    UNIQUE(team_id, player_id)
);
```

### 4.6 Player Stats Table
```sql
CREATE TABLE player_stats (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    season INTEGER,
    games_played INTEGER,
    minutes_per_game DECIMAL(5,2),
    points DECIMAL(5,2),
    rebounds DECIMAL(5,2),
    assists DECIMAL(5,2),
    steals DECIMAL(5,2),
    blocks DECIMAL(5,2),
    turnovers DECIMAL(5,2),
    field_goal_pct DECIMAL(5,3),
    free_throw_pct DECIMAL(5,3),
    three_pointers_made DECIMAL(5,2),
    stat_date DATE,
    source VARCHAR(50), -- 'ESPN', 'BASKETBALL_REFERENCE'
    UNIQUE(player_id, season, stat_date, source)
);
```

### 4.7 Projections Table
```sql
CREATE TABLE projections (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    league_id INTEGER REFERENCES leagues(id),
    season INTEGER,
    projection_type VARCHAR(50), -- 'STATISTICAL', 'ML', 'HYBRID'
    games_remaining INTEGER,
    projected_points DECIMAL(5,2),
    projected_rebounds DECIMAL(5,2),
    projected_assists DECIMAL(5,2),
    projected_steals DECIMAL(5,2),
    projected_blocks DECIMAL(5,2),
    projected_turnovers DECIMAL(5,2),
    projected_fg_pct DECIMAL(5,3),
    projected_ft_pct DECIMAL(5,3),
    projected_threes DECIMAL(5,2),
    fantasy_value DECIMAL(8,2), -- Calculated fantasy points based on league scoring
    confidence DECIMAL(5,2), -- 0-100
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, league_id, season, projection_type, created_at)
);
```

### 4.8 Trade History Table
```sql
CREATE TABLE trade_history (
    id SERIAL PRIMARY KEY,
    league_id INTEGER REFERENCES leagues(id) ON DELETE CASCADE,
    team1_id INTEGER REFERENCES teams(id),
    team2_id INTEGER REFERENCES teams(id),
    team1_players JSONB, -- Array of player IDs
    team2_players JSONB, -- Array of player IDs
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    value_differential DECIMAL(8,2),
    was_suggested BOOLEAN DEFAULT FALSE,
    was_accepted BOOLEAN
);
```

### 4.9 Waiver Recommendations Table
```sql
CREATE TABLE waiver_recommendations (
    id SERIAL PRIMARY KEY,
    league_id INTEGER REFERENCES leagues(id) ON DELETE CASCADE,
    team_id INTEGER REFERENCES teams(id),
    player_id INTEGER REFERENCES players(id),
    impact_score DECIMAL(5,2), -- 0-100
    suggested_drop_player_id INTEGER REFERENCES players(id),
    recommendation_date DATE,
    was_followed BOOLEAN,
    UNIQUE(league_id, team_id, player_id, recommendation_date)
);
```

---

## 5. API Endpoints

### 5.1 Authentication Endpoints

**POST /api/auth/register**
- Request: `{ email, password }`
- Response: `{ user_id, token }`

**POST /api/auth/login**
- Request: `{ email, password }`
- Response: `{ user_id, token, user_info }`

**POST /api/auth/logout**
- Request: `{ token }`
- Response: `{ success: true }`

**GET /api/auth/verify**
- Headers: `Authorization: Bearer <token>`
- Response: `{ valid: true, user_id }`

### 5.2 League Endpoints

**POST /api/leagues**
- Create/add a league
- Request: `{ espn_league_id, season, espn_s2, swid }`
- Response: `{ league_id, league_name, settings }`

**GET /api/leagues**
- Get all leagues for authenticated user
- Response: `[ { league_id, league_name, season, last_updated } ]`

**GET /api/leagues/:id**
- Get specific league details
- Response: `{ league_info, teams, settings }`

**POST /api/leagues/:id/refresh**
- Manually trigger data refresh
- Response: `{ success: true, updated_at }`

**DELETE /api/leagues/:id**
- Remove a league from user's account
- Response: `{ success: true }`

### 5.3 Projection Endpoints

**GET /api/leagues/:id/projections/standings**
- Get projected standings
- Response: 
```json
{
  "current_standings": [ /* teams with current records */ ],
  "projected_standings": [ /* teams with projected records */ ],
  "last_updated": "timestamp"
}
```

**GET /api/leagues/:id/projections/team/:team_id**
- Get detailed projections for a specific team
- Response: H2H or Roto specific projection data

**GET /api/leagues/:id/projections/matchup**
- H2H only: Get current/upcoming matchup projection
- Query params: `?week=<week_number>`
- Response: Category-by-category win probabilities

### 5.4 Trade Endpoints

**POST /api/leagues/:id/trades/analyze**
- Analyze a potential trade
- Request:
```json
{
  "team1_id": 1,
  "team1_players": [123, 456],
  "team2_id": 2,
  "team2_players": [789, 012]
}
```
- Response: Trade impact analysis

**GET /api/leagues/:id/trades/suggestions**
- Get AI-generated trade suggestions
- Query params: `?team_id=<team_id>`
- Response: Ranked list of suggested trades

**GET /api/leagues/:id/trades/history**
- Get past trade analyses
- Response: Array of historical trade analyses

### 5.5 Waiver Wire Endpoints

**GET /api/leagues/:id/waivers/recommendations**
- Get waiver wire pickup recommendations
- Query params: `?team_id=<team_id>&position=<position>&limit=<n>`
- Response: Ranked list of available players with impact scores

**GET /api/leagues/:id/waivers/player/:player_id**
- Get detailed analysis for specific free agent
- Response: Player stats, projections, add/drop suggestion

**GET /api/leagues/:id/waivers/streaming**
- H2H only: Get streaming recommendations for current week
- Query params: `?team_id=<team_id>`
- Response: Players with favorable schedules

### 5.6 Player Endpoints

**GET /api/players/:id**
- Get player information and stats
- Response: Player details, current stats, projections

**GET /api/players/search**
- Search for players
- Query params: `?query=<search_term>`
- Response: Array of matching players

---

## 6. Machine Learning Models

### 6.1 Pre-trained Models

**6.1.1 Counting Stats Regression Models**
- Separate models for each stat category (points, rebounds, assists, etc.)
- Features: age, experience, minutes, usage rate, team pace, position
- Algorithm: Gradient Boosting Regressor or Random Forest
- Training data: Last 5 seasons of NBA data from Basketball Reference
- Model persistence: Save as .pkl files in `backend/projections/trained_models/`

**6.1.2 Shooting Percentage Models**
- Field Goal % and Free Throw % regression models
- Features: career averages, shot volume, shot locations, defender distance
- Algorithm: Ridge Regression or Lasso
- Training data: Basketball Reference shooting splits

**6.1.3 Win Probability Classifier (H2H)**
- Predict matchup outcomes
- Features: category differentials, schedule strength, recent performance
- Algorithm: Logistic Regression or XGBoost
- Training data: Historical ESPN fantasy matchup results

**6.1.4 League Standing Predictor (Roto)**
- Predict final category rankings
- Features: current stats, projections, games remaining, team strength
- Algorithm: Ordinal Regression
- Training data: Historical Roto league final standings

### 6.2 Model Training Pipeline

**6.2.1 Data Collection**
- Scrape Basketball Reference for player stats (2019-2024 seasons)
- Collect ESPN fantasy league data (if available)
- Store in structured format for model training

**6.2.2 Feature Engineering**
- Calculate rolling averages (7-day, 15-day, 30-day)
- Per-36 minutes stats
- Usage rate, true shooting percentage
- Team-relative metrics (% of team points, rebounds, etc.)
- Schedule difficulty ratings

**6.2.3 Model Training**
- Train/validation/test split (70/15/15)
- Cross-validation for hyperparameter tuning
- Evaluate with RMSE, MAE for regression; accuracy, AUC for classification

**6.2.4 Model Evaluation Metrics**
- Projection accuracy: Compare to actual season results
- Category correlation: R² for each stat category
- Win probability calibration: Brier score
- Confidence interval coverage

### 6.3 Model Updates

**Annual Updates:**
- Retrain ML models annually before each NBA season
- Update previous season stats database for all active players
- Validate ESPN projection import pipeline

**In-Season Adjustments:**
- Mid-season model performance evaluation
- Track projection accuracy vs actual results
- Weights in tiered hybrid system are fixed by design (see Section 3.3.4)

**Tiered Weighting System:**
The primary projection approach uses the tiered weighting system documented in Section 3.3.4. This system automatically adjusts source weights based on games played, requiring no manual intervention during the season. The four-tier design ensures:
- Early season: Projections and historical data drive predictions
- Mid season: Balanced blend as sample sizes grow
- Late season: Current performance is weighted most heavily
- End of season (35+ games): 100% reliance on actual performance data

---

## 7. External Data Scraping

### 7.1 Basketball Reference Scraper

**7.1.1 Data to Scrape**
- Per-game stats for all active NBA players
- Advanced stats (usage rate, PER, true shooting %, etc.)
- Shooting splits
- Player game logs
- Team schedules and strength of schedule

**7.1.2 Scraping Strategy**
- Use BeautifulSoup4 to parse HTML tables
- Respect robots.txt and rate limiting (1-2 requests per second)
- Cache scraped data to minimize requests
- Update player data daily during NBA season
- Error handling for missing data or HTML changes

**7.1.3 Data Processing**
- Parse HTML tables into pandas DataFrames
- Clean and normalize player names (handle special characters)
- Handle inactive/injured players
- Store in database for quick access

### 7.2 ESPN API Integration (via espn-api package)

**7.2.1 Data to Fetch**
- League settings and scoring rules
- Team rosters and current standings
- Player stats (season totals, recent games)
- ESPN's own player projections
- Free agent pool
- Historical matchup results
- Trade and waiver transactions

**7.2.2 API Usage**
- Initialize ESPN client with league ID, year, espn_s2, and SWID
- Handle authentication errors gracefully
- Implement retry logic for failed requests
- Cache responses with appropriate TTL
- Rate limiting to avoid ESPN throttling

---

## 8. User Interface Design

### 8.1 Page Layout & Navigation

**8.1.1 Navbar (All Pages)**
- Logo/App name (links to dashboard)
- League selector dropdown (if multiple leagues)
- Refresh button with last updated timestamp
- User profile icon with dropdown:
  - Settings
  - Add New League
  - Logout

**8.1.2 Sidebar Navigation (Desktop)**
- Dashboard
- Projections
- Trade Analyzer
- Waiver Wire
- Matchups (H2H only)
- Settings

**8.1.3 Mobile Navigation**
- Hamburger menu for navigation
- Responsive design for all components
- Touch-friendly buttons and interactions

### 8.2 Page Designs

**8.2.1 Login/Registration Page**
- Clean, centered form
- Email and password inputs
- "Remember me" checkbox
- Links to switch between login/register
- Form validation with error messages

**8.2.2 League Setup Page**
- Step-by-step wizard:
  1. Enter ESPN league URL
  2. Instructions for finding ESPN_S2 and SWID cookies (with screenshots)
  3. Paste cookies
  4. Validate and import league data
- Progress indicator
- "Need Help?" expandable section with detailed instructions

**8.2.3 Dashboard**
- **Header Section:**
  - League name and format (H2H/Roto)
  - Current week (H2H) or current date
  - Last updated timestamp
  - Manual refresh button
  
- **Main Content (3-column layout for desktop):**
  - **Left Column: Current Standings**
    - Table with team names, records, current rank
    - Highlight user's team
  
  - **Middle Column: Projected Standings**
    - Table with team names, projected records, projected rank
    - Up/down arrows showing movement from current
    - Playoff probability (H2H) or win probability (Roto)
  
  - **Right Column: Quick Insights**
    - Top waiver wire targets (top 3)
    - Trade opportunities (top 2)
    - Upcoming matchup preview (H2H)
    - Category strengths/weaknesses

- **Bottom Section: Visualizations**
  - **H2H:** Line chart of win probability over time
  - **Roto:** Bar chart of category rankings (current vs projected)

**8.2.4 Projections Page**
- Tab navigation:
  - **Season Projections:** Full standings projection
  - **Playoff Odds** (H2H): Bracket simulation, playoff matchup probabilities
  - **Category Analysis** (Roto): Detailed category rankings and gaps

- Interactive charts and tables
- Export to CSV option

**8.2.5 Trade Analyzer Page**
- **Trade Builder Section:**
  - Two-panel interface (Your Team vs Trade Partner)
  - Dropdown to select trade partner
  - Roster lists with checkboxes to select players
  - Visual indicator of selected players
  - "Analyze Trade" button

- **Analysis Results Section:**
  - Side-by-side comparison:
    - Pre-trade team values
    - Post-trade team values
    - Net change in category stats
  - Win probability impact
  - Fairness meter (visual indicator)
  - "Trade Recommendation" summary

- **Suggested Trades Section:**
  - Collapsible list of AI-generated trade ideas
  - Filter by trade partner
  - Each suggestion shows:
    - Players involved
    - Expected impact
    - "Analyze This Trade" button

**8.2.6 Waiver Wire Page**
- **Filters Bar:**
  - Position filter (All, PG, SG, SF, PF, C)
  - Sort by: Impact Score, Projected Points, Recent Performance
  - Search bar for player names

- **Recommendations List:**
  - Cards for each recommended player:
    - Player name, position, NBA team
    - Impact score (0-100) with color coding
    - Key stats (last 15 games)
    - Rest-of-season projections
    - "Suggested Drop" player (if applicable)
    - "Add Player" button (links to ESPN, future)

- **Player Detail Modal:**
  - Expanded stats and projections
  - Recent game log
  - Season chart (points over time)
  - Injury status and news
  - Schedule analysis

**8.2.7 Matchups Page (H2H Only)**
- **Current Week Matchup:**
  - Opponent team name and record
  - Category-by-category projection table
  - Win probability for each category
  - Overall matchup win probability
  - Suggested lineup optimizations
  - Streaming opportunities for the week

- **Upcoming Matchups:**
  - Calendar view of future matchups
  - Difficulty ratings for each opponent
  - Schedule analysis (games per matchup)

**8.2.8 Settings Page**
- **Account Settings:**
  - Email address
  - Change password
  - Delete account

- **League Settings:**
  - List of connected leagues
  - Add new league button
  - Remove league option
  - Edit ESPN cookies (if expired)

- **Projection Settings (per league):**
  - Projection method toggle (Adaptive vs Flat Rate)
  - Flat rate value slider (when flat rate selected)
  - Preview of how settings affect projections

- **Preferences:**
  - Refresh schedule time
  - Email notifications toggle (future)
  - Display preferences (dark mode, compact view)

- **Help & Documentation:**
  - Link to user guide
  - Contact support (future)

### 8.3 UI/UX Principles

- **Responsive Design:** Mobile-first approach, works on all screen sizes
- **Loading States:** Skeleton screens and spinners for data fetching
- **Error Handling:** User-friendly error messages with suggested actions
- **Accessibility:** WCAG 2.1 AA compliance, keyboard navigation, screen reader support
- **Performance:** Lazy loading for images and components, optimized API calls
- **Consistency:** Unified color scheme, typography, and component styling

---

## 9. Development Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up project structure and Git repository
- [ ] Configure Flask backend with SQLAlchemy
- [ ] Set up SQLite database with schema
- [ ] Create React frontend with basic routing
- [ ] Implement user authentication (register, login, logout)
- [ ] Build ESPN cookie setup workflow

### Phase 2: Data Integration (Weeks 3-4)
- [ ] Integrate espn-api package for league data
- [ ] Build ESPN client service with error handling
- [ ] Implement data caching layer
- [ ] Create Basketball Reference scraper
- [ ] Build data models and database CRUD operations
- [ ] Implement daily refresh scheduler

### Phase 3: Projection Engine (Weeks 5-7)
- [ ] Collect and prepare historical training data
- [ ] Train ML models for counting stats
- [ ] Train shooting percentage models
- [ ] Build statistical projection component
- [ ] Implement hybrid projection engine
- [ ] Create H2H matchup simulator
- [ ] Build Roto category predictor
- [ ] Test projection accuracy

### Phase 4: Core Features (Weeks 8-10)
- [ ] Build dashboard with standings and projections
- [ ] Implement trade analyzer logic
- [ ] Create waiver wire recommendation algorithm
- [ ] Build H2H matchup analysis
- [ ] Implement manual refresh functionality
- [ ] Add data visualization components

### Phase 5: Frontend Development (Weeks 11-13)
- [ ] Build all React components for dashboard
- [ ] Create trade analyzer UI
- [ ] Design waiver wire recommendations interface
- [ ] Build matchups page (H2H)
- [ ] Implement projections page with charts
- [ ] Add settings and profile management
- [ ] Implement responsive design

### Phase 6: Testing & Refinement (Weeks 14-15)
- [ ] Write unit tests for backend services
- [ ] Write integration tests for API endpoints
- [ ] Test with real ESPN leagues (H2H and Roto)
- [ ] User acceptance testing
- [ ] Bug fixes and performance optimization
- [ ] Code review and refactoring

### Phase 7: Documentation & Deployment (Week 16)
- [ ] Write API documentation
- [ ] Create user guide with screenshots
- [ ] Document deployment process
- [ ] Set up local development environment
- [ ] Prepare for future production deployment
- [ ] Final testing and launch

---

## 10. Testing Strategy

### 10.1 Backend Testing

**Unit Tests:**
- Test ESPN API client methods
- Test projection engine components (statistical, ML, hybrid)
- Test trade analyzer logic
- Test waiver recommender algorithms
- Test database models and CRUD operations

**Integration Tests:**
- Test API endpoints with mock data
- Test end-to-end data flow (ESPN → DB → Projections → API)
- Test authentication flow
- Test scheduled tasks (daily refresh)

**Test Coverage Goal:** 80%+ code coverage

### 10.2 Frontend Testing

**Component Tests:**
- Test React components in isolation
- Test user interactions (button clicks, form submissions)
- Test conditional rendering
- Test API service calls

**End-to-End Tests:**
- Test complete user workflows:
  - Registration and login
  - Adding a league
  - Viewing projections
  - Analyzing trades
  - Viewing waiver recommendations

**Tools:**
- Jest and React Testing Library
- Cypress for E2E tests

### 10.3 Manual Testing

- Test with real ESPN leagues (both H2H and Roto)
- Test on different browsers (Chrome, Firefox, Safari)
- Test on mobile devices (iOS and Android)
- Test edge cases (empty leagues, mid-season additions, etc.)
- Performance testing (large leagues, many players)

### 10.4 Model Validation

- Validate projection accuracy against actual season outcomes
- Test on historical seasons (2022-2023, 2023-2024)
- Calculate error metrics (RMSE, MAE) for each stat category
- Validate win probability calibration with Brier score
- Compare to ESPN's own projections

---

## 11. Security & Privacy

### 11.1 Authentication Security
- Password hashing with bcrypt (minimum 12 rounds)
- Secure session management with Flask-Login
- CSRF protection on all forms
- Rate limiting on login attempts
- Secure cookie configuration (HttpOnly, Secure, SameSite)

### 11.2 Data Protection
- ESPN cookies encrypted at rest in database
- No storage of ESPN passwords
- User data isolated by user_id
- Parameterized SQL queries to prevent SQL injection
- Input validation and sanitization

### 11.3 API Security
- JWT or session-based authentication for API requests
- Authorization checks on all protected endpoints
- CORS configuration for frontend-backend communication
- Rate limiting on API endpoints

### 11.4 Privacy
- Clear privacy policy about data usage
- ESPN data used only for user's own analysis
- No sharing of user data with third parties
- Option to delete account and all associated data
- Compliance with data protection regulations (GDPR, CCPA)

### 11.5 ESPN Terms of Service
- Respect ESPN's terms of service
- Avoid aggressive scraping or API abuse
- Implement rate limiting and caching
- Clear disclaimers that app is not affiliated with ESPN

---

## 12. Performance Considerations

### 12.1 Backend Performance
- Database indexing on frequently queried columns (league_id, player_id, etc.)
- Connection pooling for database connections
- Caching of ESPN API responses (Redis or in-memory cache)
- Asynchronous tasks for long-running operations (projection calculations)
- Query optimization to reduce N+1 queries

### 12.2 Frontend Performance
- Code splitting and lazy loading of routes
- Memoization of expensive calculations
- Debouncing for search inputs
- Virtualization for long lists (player lists)
- Image optimization and lazy loading

### 12.3 Scalability (Future)
- Horizontal scaling with load balancer
- Separate worker processes for background jobs
- CDN for static assets
- Database read replicas for heavy read operations

---

## 13. Future Enhancements

### 13.1 Near-term (Post-MVP)
- Email notifications for key events
- Export reports to PDF
- Mobile app (React Native)
- Support for Yahoo and Sleeper fantasy platforms
- Auction draft assistant
- In-season tournament optimizer

### 13.2 Long-term
- Social features (share projections, trade ideas)
- League chat integration
- Advanced visualizations (heat maps, network graphs)
- Machine learning model marketplace (users can try different models)
- Real-time game day updates and alerts
- Integration with sports betting odds
- Voice assistant integration (Alexa, Google Assistant)
- Browser extension for ESPN website

---

## 14. Success Metrics

### 14.1 Technical Metrics
- Projection accuracy (RMSE < 2.0 for counting stats)
- API response time (< 500ms for dashboard load)
- System uptime (> 99% during NBA season)
- Test coverage (> 80%)

### 14.2 User Metrics (Future)
- User retention rate
- Daily/weekly active users
- Average session duration
- Feature adoption rates (trades analyzed, waivers followed)
- User satisfaction (NPS score)

### 14.3 Product Metrics
- Accuracy of trade recommendations (% that improve team)
- Waiver recommendations success rate (% that outperform dropped player)
- Playoff prediction accuracy (% of teams correctly predicted)

---

## 15. Risks & Mitigation

### 15.1 Technical Risks

**Risk:** ESPN API changes or rate limiting
- **Mitigation:** Robust error handling, caching, fallback to web scraping if needed

**Risk:** Basketball Reference blocks scraping
- **Mitigation:** Respect robots.txt, implement polite scraping, have backup data sources

**Risk:** Model inaccuracy
- **Mitigation:** Continuous validation, confidence intervals, multiple model ensemble

**Risk:** Database performance issues
- **Mitigation:** Proper indexing, query optimization, caching layer

### 15.2 Legal Risks

**Risk:** ESPN terms of service violation
- **Mitigation:** Review TOS carefully, use official API where possible, add disclaimers

**Risk:** Copyright issues with scraped data
- **Mitigation:** Only scrape publicly available data, attribute sources properly

### 15.3 User Experience Risks

**Risk:** Complex setup process (ESPN cookies)
- **Mitigation:** Clear documentation with screenshots, video tutorial

**Risk:** Inaccurate recommendations damage trust
- **Mitigation:** Show confidence levels, explain methodology, track accuracy

---

## 16. Documentation Requirements

### 16.1 Developer Documentation
- README with setup instructions
- API documentation (endpoints, request/response formats)
- Database schema documentation
- Code comments for complex algorithms
- Contributing guidelines

### 16.2 User Documentation
- User guide for setting up ESPN connection
- FAQ section
- Video tutorials for key features
- Troubleshooting guide
- Feature glossary (explaining projection terminology)

### 16.3 Deployment Documentation
- Local development setup
- Environment variables configuration
- Database migration guide
- Production deployment checklist
- Backup and recovery procedures

---

## 17. Conclusion

The Fantasy Basketball Optimizer is a comprehensive web application that combines data science, machine learning, and web development to provide actionable insights for fantasy basketball managers. By integrating with ESPN Fantasy Basketball and leveraging external data sources, the app delivers accurate projections, intelligent trade recommendations, and optimized waiver wire suggestions.

The project is structured for iterative development with clear phases, allowing for incremental feature delivery and continuous testing. With a focus on user experience, performance, and accuracy, this app has the potential to become an essential tool for serious fantasy basketball players.

**Next Steps:**
1. Review and approve this PRD
2. Set up development environment
3. Begin Phase 1 development (Foundation)
4. Regular check-ins to track progress and adjust priorities

---

**Document Version:** 1.2
**Last Updated:** February 19, 2026
**Changes:** Added Z-Score Value System (3.3.11), simplified IR Drop Optimizer (3.3.8)
**Author:** Milo (with Claude)
**Status:** Active Development

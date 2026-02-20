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

### 8. Trade Analyzer
- Input: Multi-player trades (any size, 2 teams only)
- Output: Pre/post trade comparison, category value changes, win probability impact, fairness assessment
- AI-generated trade suggestions based on team needs

### 9. Waiver Wire Recommendations
- Ranked list of available free agents
- Impact score (0-100) for each player
- Add/drop suggestions
- Streaming recommendations for H2H leagues (favorable weekly schedules)

### 10. Dashboard
- Current vs projected standings
- Playoff/win probability visualization
- Quick insights (top waiver targets, trade opportunities)
- Category strengths/weaknesses

---

---

## Key Bug Fixes (February 2026)

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

**Phase:** Phase 3 - Projection Engine (In Progress)
**Completed:**
- Project structure and Flask backend
- SQLite database with migrations (Flask-Migrate)
- ESPN integration with espn-api
- Injury handling from ESPN kona_playercard endpoint
- Projection method settings (adaptive/flat rate)
- Schedule-based remaining games calculation
- Start limit optimizer with projected_games enforcement

**Next Steps:**
1. Complete hybrid projection engine
2. Build trade analyzer logic
3. Create waiver wire recommendation algorithm
4. Build React frontend components

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

**Last Updated:** February 19, 2026
**Version:** 1.3 (Z-Score Value System & IR Drop Optimizer Simplification)

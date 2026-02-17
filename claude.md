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

### 5. IR Player Handling & Return Projections
- **Detection:** Players in IR slot identified by `lineupSlotId == 13` from ESPN API
- **Return Projection:** Uses ESPN's expected return date
- **RotoDropScenario Analysis:** When IR player returns:
  - Calculates marginal value for each rostered player
  - Simulates roster from return date forward
  - Recommends optimal drop candidate with least Roto point loss
  - Shows projected standings impact

### 6. Day-by-Day Start Limit Optimization (Roto)
For leagues with position start limits (e.g., 82 games per position):
- **Daily Simulation:** For each remaining day, identifies which players have games
- **Starter Assignment:** Assigns highest-value eligible player to each slot
- **Limit Enforcement:** Tracks `starts_used` per position, stops when limit reached
- **Conflict Resolution:** Prioritizes higher-value players for limited slots
- **Output:** Realistic projections based on actual games player will start

### 7. Trade Analyzer
- Input: Multi-player trades (any size, 2 teams only)
- Output: Pre/post trade comparison, category value changes, win probability impact, fairness assessment
- AI-generated trade suggestions based on team needs

### 8. Waiver Wire Recommendations
- Ranked list of available free agents
- Impact score (0-100) for each player
- Add/drop suggestions
- Streaming recommendations for H2H leagues (favorable weekly schedules)

### 9. Dashboard
- Current vs projected standings
- Playoff/win probability visualization
- Quick insights (top waiver targets, trade opportunities)
- Category strengths/weaknesses

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

### Phase 1: Foundation ✓ (YOU ARE HERE)
- Set up project structure
- Configure Flask backend with SQLAlchemy
- Set up SQLite database with schema
- Create React frontend with basic routing
- Implement user authentication
- Build ESPN cookie setup workflow

### Phase 2: Data Integration
- Integrate espn-api package
- Build ESPN client service
- Implement data caching
- Create Basketball Reference scraper
- Build database CRUD operations
- Implement daily refresh scheduler

### Phase 3: Projection Engine
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

# Database migrations (when using Flask-Migrate)
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

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

**Phase:** Phase 1 - Foundation
**Next Steps:**
1. Set up initial project structure
2. Create backend Flask app with basic routing
3. Set up SQLite database with schema
4. Implement user authentication
5. Create basic React frontend

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

**Last Updated:** February 11, 2026
**Version:** 1.1

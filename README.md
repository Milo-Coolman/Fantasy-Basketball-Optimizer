# Fantasy Basketball Optimizer

A web application that integrates with ESPN Fantasy Basketball to provide advanced analytics, projections, and recommendations. The app analyzes league data to project end-of-season standings, recommend trades, and suggest waiver wire acquisitions.

## Features

- **ESPN Integration** - Connect your ESPN Fantasy Basketball league using cookies
- **Projection Engine** - Hybrid ML + statistical model for player and team projections
- **Trade Analyzer** - Evaluate trade impact with pre/post comparison and fairness assessment
- **Waiver Wire Recommendations** - Ranked free agent suggestions with impact scores
- **Dashboard** - Current vs. projected standings, playoff probabilities, and insights
- **H2H & Roto Support** - Works with Head-to-Head Categories and Rotisserie leagues

## Tech Stack

**Backend:** Python 3.10+, Flask, SQLAlchemy, scikit-learn, pandas
**Frontend:** React 18+, React Router, Axios, Chart.js/Recharts, Tailwind CSS
**Database:** SQLite (development), PostgreSQL (production)
**External:** espn-api, Basketball Reference (scraping)

## Project Structure

```
fantasy-basketball-optimizer/
├── backend/                 # Flask API and services
│   ├── api/                 # API endpoints
│   ├── services/            # ESPN client, caching, scheduler
│   ├── projections/         # ML and statistical models
│   ├── analyzers/           # Trade and waiver analysis
│   ├── scrapers/            # Basketball Reference scraper
│   ├── utils/               # Helper functions
│   └── tests/               # Backend tests
├── frontend/                # React application
│   ├── public/              # Static files
│   └── src/                 # React components and services
├── database/                # Schema and migrations
└── docs/                    # Documentation
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your settings

# Run Flask app
flask run
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm start
```

## ESPN Setup

To connect your ESPN Fantasy Basketball league:

1. Log in to ESPN Fantasy Basketball in your browser
2. Extract your `ESPN_S2` and `SWID` cookies (see docs/USER_GUIDE.md)
3. Enter your league URL and cookies in the app

## Development

### Running Tests

```bash
# Backend tests
python -m pytest backend/tests/

# Frontend tests
cd frontend && npm test
```

### Code Style

- Python: PEP 8
- JavaScript: ESLint with React rules

## Documentation

- [API Documentation](docs/API.md)
- [User Guide](docs/USER_GUIDE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## License

This project is for personal use only. Not affiliated with ESPN.

## Author

Milo - [GitHub](https://github.com/milo)

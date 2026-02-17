import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fetchDashboard, refreshLeagueData } from '../services/api';
import StandingsTable from './StandingsTable';
import ProjectionsChart from './ProjectionsChart';
import CategoryComparisonChart from './CategoryComparisonChart';
import QuickInsights from './QuickInsights';
import StartLimitsInfo from './StartLimitsInfo';

/**
 * LeagueDashboard - Comprehensive league view with standings, projections, and insights
 */
function LeagueDashboard() {
  const { leagueId } = useParams();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  /**
   * Fetch dashboard data from backend
   */
  const loadDashboard = useCallback(async (showRefresh = false) => {
    if (showRefresh) {
      setRefreshing(true);
    }
    try {
      const data = await fetchDashboard(leagueId);
      setDashboardData(data);
      setError('');
    } catch (err) {
      console.error('Error fetching dashboard:', err);
      setError(err.response?.data?.error || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [leagueId]);

  /**
   * Load data on component mount
   */
  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  /**
   * Handle manual refresh - fetches fresh data from ESPN
   */
  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      setError('');

      // First trigger ESPN data refresh
      await refreshLeagueData(leagueId);

      // Then fetch updated dashboard
      await loadDashboard(true);
    } catch (err) {
      console.error('Error refreshing data:', err);
      setError(err.response?.data?.error || 'Failed to refresh data');
      setRefreshing(false);
    }
  };

  /**
   * Format date for display
   */
  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  /**
   * Calculate current fantasy week
   */
  const getCurrentWeek = () => {
    const seasonStart = new Date('2024-10-22');
    const now = new Date();
    const weeksDiff = Math.floor((now - seasonStart) / (7 * 24 * 60 * 60 * 1000));
    return Math.max(1, weeksDiff + 1);
  };

  // Loading state
  if (loading) {
    return (
      <div className="league-dashboard">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  // Error state with no data
  if (error && !dashboardData) {
    return (
      <div className="league-dashboard">
        <div className="alert alert-error">{error}</div>
        <Link to="/dashboard" className="btn btn-secondary">
          Back to Leagues
        </Link>
      </div>
    );
  }

  // Extract data from dashboard response
  const {
    league,
    current_standings = [],
    projected_standings = [],
    user_team,
    insights = {},
    last_updated,
    scoring_categories = [],
    is_roto = false,
    category_data = {},
    start_limits = {},
  } = dashboardData || {};

  const isH2H = league?.league_type?.includes('H2H');
  const isRoto = is_roto || league?.league_type === 'ROTO';
  const userTeamId = user_team?.team_id;

  // Convert scoring_categories from backend format to frontend format
  const rotoCategories = scoring_categories.length > 0
    ? scoring_categories.map(cat => ({
        key: cat.key,
        label: cat.label,
        abbr: cat.abbr,
      }))
    : null;

  // Prepare standings data for StandingsTable component
  const currentStandingsData = current_standings.map(team => ({
    ...team,
    rank: team.rank || team.current_rank,
    is_user_team: team.team_id === userTeamId || team.is_user_team,
  }));

  const projectedStandingsData = projected_standings.map(team => ({
    ...team,
    rank: team.projected_rank || team.projected_standing,
    current_rank: team.current_rank || team.current_standing || team.rank,
    projected_rank: team.projected_rank || team.projected_standing,
    projected_category_ranks: team.projected_category_ranks || team.category_ranks,
    category_ranks: team.category_ranks,
    roto_points: team.projected_roto_points || team.roto_points,
    total_points: team.projected_total_points || team.total_points,
    is_user_team: team.team_id === userTeamId || team.espn_team_id === userTeamId || team.is_user_team,
  }));

  // Prepare insights data for QuickInsights component
  const waiverTargets = insights.waiver_targets || [];
  const tradeOpportunities = insights.trade_opportunities || [];
  const categoryAnalysis = insights.category_analysis || {};

  // Generate category movements for Roto leagues
  const generateCategoryMovements = () => {
    if (!isRoto || !user_team) return [];

    const userCurrentStanding = currentStandingsData.find(
      t => t.team_id === userTeamId || t.is_user_team
    );
    const userProjectedStanding = projectedStandingsData.find(
      t => t.team_id === userTeamId || t.espn_team_id === userTeamId || t.is_user_team
    );

    if (!userCurrentStanding || !userProjectedStanding) return [];

    const currentRanks = userCurrentStanding.category_ranks || {};
    const projectedRanks = userProjectedStanding.projected_category_ranks ||
                          userProjectedStanding.category_ranks || {};

    // Use dynamic categories from API - if not available, return empty array
    if (!rotoCategories || rotoCategories.length === 0) {
      return [];
    }

    return rotoCategories
      .map(cat => {
        const currentRank = currentRanks[cat.key] || 0;
        const projectedRank = projectedRanks[cat.key] || currentRank;
        const movement = currentRank - projectedRank; // Positive = improving (lower rank is better)

        return {
          category: cat.abbr || cat.label || cat.key.toUpperCase(),
          currentRank,
          projectedRank,
          movement,
        };
      })
      .filter(m => m.currentRank > 0); // Only include categories with data
  };

  const categoryMovements = generateCategoryMovements();

  // Get league type display name
  const getLeagueTypeDisplay = () => {
    if (isRoto) return 'Rotisserie';
    if (league?.league_type?.includes('H2H_CATEGORY')) return 'Head-to-Head Categories';
    if (league?.league_type?.includes('H2H_POINTS')) return 'Head-to-Head Points';
    return league?.league_type || 'Fantasy';
  };

  return (
    <div className={`league-dashboard ${isRoto ? 'roto-league' : 'h2h-league'}`}>
      {/* League Type Banner */}
      <div className={`league-type-banner ${isRoto ? 'roto' : 'h2h'}`}>
        <div className="banner-content">
          <span className="banner-icon">
            {isRoto ? (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 6v6l4 2" />
              </svg>
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 00-3-3.87" />
                <path d="M16 3.13a4 4 0 010 7.75" />
              </svg>
            )}
          </span>
          <span className="banner-text">
            League Type: <strong>{getLeagueTypeDisplay()}</strong>
          </span>
          {isRoto && (
            <span className="banner-hint">
              Season-long category accumulation scoring
            </span>
          )}
        </div>
      </div>

      {/* Header */}
      <header className="league-dashboard-header">
        <div className="header-left">
          <Link to="/dashboard" className="back-link">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            Back
          </Link>
          <h1>{league?.league_name || 'League Dashboard'}</h1>
          <div className="header-meta">
            <span className={`league-type-badge ${isRoto ? 'roto' : 'h2h'}`}>
              {isRoto ? 'ROTO' : league?.league_type}
            </span>
            <span className="week-badge">Week {getCurrentWeek()}</span>
            <span className="date-text">
              {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
            </span>
          </div>
        </div>
        <div className="header-right">
          <span className="last-updated">
            Updated: {formatDate(last_updated)}
          </span>
          <button
            className="btn btn-primary refresh-btn"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            {refreshing ? (
              <>
                <span className="btn-spinner"></span>
                Refreshing...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M23 4v6h-6M1 20v-6h6" />
                  <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
                </svg>
                Refresh
              </>
            )}
          </button>
        </div>
      </header>

      {/* Error banner (non-blocking) */}
      {error && <div className="alert alert-error">{error}</div>}

      {/* User Team Highlight */}
      {user_team && (
        <UserTeamHighlight
          userTeam={user_team}
          isH2H={isH2H}
          isRoto={isRoto}
          categoryMovements={categoryMovements}
        />
      )}

      {/* Start Limits Info - Only show for Roto leagues when enabled */}
      {isRoto && start_limits?.enabled && (
        <div className="start-limits-section">
          <StartLimitsInfo
            startLimits={start_limits}
            showIRReturns={true}
            compact={false}
            title="Position Start Limits"
          />
        </div>
      )}

      {/* Main Content - Three Column Layout */}
      <div className="dashboard-columns">
        {/* Current Standings */}
        <div className="dashboard-column">
          <StandingsTable
            standings={currentStandingsData}
            title="Current Standings"
            leagueType={league?.league_type || 'H2H'}
            showOwner={false}
            showProjected={false}
            showProbability={false}
            userTeamId={userTeamId}
            categories={isRoto ? rotoCategories : null}
          />
        </div>

        {/* Projected Standings */}
        <div className="dashboard-column">
          <StandingsTable
            standings={projectedStandingsData}
            title={isRoto ? 'Projected Category Ranks' : 'Projected Standings'}
            leagueType={league?.league_type || 'H2H'}
            showOwner={false}
            showProjected={true}
            showProbability={!isRoto}
            probabilityLabel={isH2H ? 'Playoff %' : 'Win %'}
            userTeamId={userTeamId}
            categories={isRoto ? rotoCategories : null}
            startLimitsEnabled={start_limits?.enabled || false}
            irReturns={start_limits?.ir_players || []}
          />
        </div>

        {/* Quick Insights */}
        <div className="dashboard-column">
          <QuickInsights
            title={isRoto ? 'Roto Insights' : 'Quick Insights'}
            waiverTargets={waiverTargets}
            tradeOpportunities={tradeOpportunities}
            categoryAnalysis={categoryAnalysis}
            categoryMovements={categoryMovements}
            isRoto={isRoto}
            showMovements={isRoto}
            compact={true}
          />
        </div>
      </div>

      {/* Visualizations Section */}
      <div className="dashboard-visualizations">
        {isRoto ? (
          <CategoryComparisonChart
            teams={projected_standings}
            currentStandings={currentStandingsData}
            projectedStandings={projectedStandingsData}
            categoryData={category_data}
            scoringCategories={rotoCategories}
            userTeamId={userTeamId}
            title="Current vs Projected Category Points"
            startLimitsEnabled={start_limits?.enabled || false}
          />
        ) : (
          <ProjectionsChart
            leagueType={league?.league_type || 'H2H'}
            teams={projected_standings}
            userTeamId={userTeamId}
            categoryData={categoryAnalysis?.rankings}
            title="Playoff Probability Trends"
          />
        )}
      </div>
    </div>
  );
}

/**
 * User Team Highlight Component
 */
function UserTeamHighlight({ userTeam, isH2H, isRoto, categoryMovements = [] }) {
  const movement = userTeam.movement || (userTeam.current_rank - userTeam.projected_rank);

  // Calculate category summary for Roto
  const improvingCount = categoryMovements.filter(m => m.movement > 0).length;
  const decliningCount = categoryMovements.filter(m => m.movement < 0).length;

  return (
    <div className={`user-team-highlight ${isRoto ? 'roto' : ''}`}>
      <div className="highlight-content">
        <h3>Your Team: {userTeam.team_name}</h3>
        <div className="highlight-stats">
          <div className="stat-block">
            <span className="stat-label">Current Rank</span>
            <span className="stat-value">{userTeam.current_rank}</span>
          </div>
          <div className="stat-block">
            <span className="stat-label">Projected Rank</span>
            <span className="stat-value">{userTeam.projected_rank}</span>
          </div>
          <div className="stat-block movement">
            <span className="stat-label">Movement</span>
            <span className={`stat-value ${movement > 0 ? 'positive' : movement < 0 ? 'negative' : ''}`}>
              {movement > 0 ? '+' : ''}{movement}
              {movement !== 0 && (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  {movement > 0 ? (
                    <path d="M12 4l-8 8h5v8h6v-8h5z" />
                  ) : (
                    <path d="M12 20l8-8h-5V4H9v8H4z" />
                  )}
                </svg>
              )}
            </span>
          </div>

          {/* Roto-specific: Show Roto Points and Category Summary */}
          {isRoto && (
            <>
              {userTeam.roto_points !== undefined && (
                <div className="stat-block">
                  <span className="stat-label">Roto Points</span>
                  <span className="stat-value">{userTeam.roto_points}</span>
                </div>
              )}
              {categoryMovements.length > 0 && (
                <div className="stat-block category-summary-block">
                  <span className="stat-label">Categories</span>
                  <span className="stat-value category-summary-value">
                    <span className="improving-count" title="Categories improving">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 4l-8 8h5v8h6v-8h5z" />
                      </svg>
                      {improvingCount}
                    </span>
                    <span className="declining-count" title="Categories declining">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 20l8-8h-5V4H9v8H4z" />
                      </svg>
                      {decliningCount}
                    </span>
                  </span>
                </div>
              )}
            </>
          )}

          {/* H2H-specific: Show Win Probability */}
          {!isRoto && userTeam.win_probability !== undefined && (
            <div className="stat-block">
              <span className="stat-label">{isH2H ? 'Playoff Prob' : 'Win Prob'}</span>
              <span className="stat-value">{Math.round(userTeam.win_probability * 100)}%</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default LeagueDashboard;

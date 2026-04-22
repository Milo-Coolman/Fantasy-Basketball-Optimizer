import React, { useState, useEffect, useCallback } from 'react';
import { fetchSeasonRecap } from '../services/api';

/**
 * SeasonRecap - End-of-season summary with MVP and final standings
 */
function SeasonRecap({ leagueId }) {
  const [recapData, setRecapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /**
   * Fetch season recap data
   */
  const loadRecap = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchSeasonRecap(leagueId);
      setRecapData(data);
    } catch (err) {
      console.error('Error fetching season recap:', err);
      setError(err.response?.data?.error || 'Failed to load season recap');
    } finally {
      setLoading(false);
    }
  }, [leagueId]);

  useEffect(() => {
    loadRecap();
  }, [loadRecap]);

  // Loading state
  if (loading) {
    return (
      <div className="season-recap-container">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading season recap...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="season-recap-container">
        <div className="alert alert-error">{error}</div>
        <button className="btn btn-primary" onClick={loadRecap}>
          Retry
        </button>
      </div>
    );
  }

  // Season still in progress
  if (recapData?.season_status === 'in_progress') {
    return (
      <div className="season-recap-container">
        <div className="season-in-progress">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 6v6l4 2" />
          </svg>
          <h2>Season In Progress</h2>
          <p>The {recapData.season} season is still active. Check back after the season ends for your recap.</p>
        </div>
      </div>
    );
  }

  const { final_standings, season_mvp, user_team_recap, category_leaders, league, scoring_categories } = recapData || {};

  return (
    <div className="season-recap-container">
      {/* Header */}
      <div className="recap-header">
        <div className="trophy-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C9.24 2 7 4.24 7 7H5C3.34 7 2 8.34 2 10C2 11.66 3.34 13 5 13H7V14C7 16.76 9.24 19 12 19C14.76 19 17 16.76 17 14V13H19C20.66 13 22 11.66 22 10C22 8.34 20.66 7 19 7H17C17 4.24 14.76 2 12 2ZM5 11C4.45 11 4 10.55 4 10C4 9.45 4.45 9 5 9H7V11H5ZM15 14C15 15.66 13.66 17 12 17C10.34 17 9 15.66 9 14V7C9 5.34 10.34 4 12 4C13.66 4 15 5.34 15 7V14ZM19 11H17V9H19C19.55 9 20 9.45 20 10C20 10.55 19.55 11 19 11Z"/>
            <path d="M12 19V22M8 22H16" strokeWidth="2" stroke="currentColor" fill="none"/>
          </svg>
        </div>
        <h1>{league?.season} Season Recap</h1>
        <p className="recap-subtitle">{league?.name}</p>
      </div>

      {/* MVP Card */}
      {season_mvp && (
        <div className="mvp-card">
          <div className="mvp-badge">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
            </svg>
            <span>SEASON MVP</span>
          </div>
          <div className="mvp-content">
            <div className="mvp-main">
              <h2 className="mvp-name">{season_mvp.name}</h2>
              <div className="mvp-details">
                <span className="mvp-team">{season_mvp.team}</span>
                <span className="mvp-positions">{season_mvp.positions?.join(' / ')}</span>
              </div>
            </div>
            <div className="mvp-z-score">
              <span className="z-label">Total Z-Score</span>
              <span className="z-value">{season_mvp.total_z_score?.toFixed(2)}</span>
            </div>
          </div>
          <div className="mvp-stats">
            <h4>Per Game Stats</h4>
            <div className="stats-grid">
              {scoring_categories && scoring_categories.map((cat) => {
                const statKey = mapCategoryToStat(cat);
                const value = season_mvp.per_game_stats?.[statKey];
                return (
                  <div key={cat} className="stat-item">
                    <span className="stat-label">{cat}</span>
                    <span className="stat-value">
                      {typeof value === 'number'
                        ? (cat === 'FG%' || cat === 'FT%'
                            ? (value * 100).toFixed(1) + '%'
                            : value.toFixed(1))
                        : value ?? '-'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="mvp-games">
            <span>{season_mvp.games_played} games played</span>
          </div>
        </div>
      )}

      {/* User Team Recap */}
      {user_team_recap && (
        <div className="user-recap-card">
          <h3>Your Team: {user_team_recap.team_name}</h3>
          <div className="recap-stats">
            <div className="recap-stat">
              <span className="stat-value large">{getOrdinal(user_team_recap.final_rank)}</span>
              <span className="stat-label">Final Rank</span>
            </div>
            <div className="recap-stat">
              <span className="stat-value">{user_team_recap.total_points?.toFixed(1)}</span>
              <span className="stat-label">Total Points</span>
            </div>
            {user_team_recap.best_category && (
              <div className="recap-stat best">
                <span className="stat-value">{user_team_recap.best_category.name}</span>
                <span className="stat-label">Best Category (#{user_team_recap.best_category.rank})</span>
              </div>
            )}
            {user_team_recap.worst_category && (
              <div className="recap-stat worst">
                <span className="stat-value">{user_team_recap.worst_category.name}</span>
                <span className="stat-label">Worst Category (#{user_team_recap.worst_category.rank})</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Final Standings */}
      <div className="final-standings-card">
        <h3>Final Standings</h3>
        <table className="standings-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Team</th>
              <th>Points</th>
            </tr>
          </thead>
          <tbody>
            {final_standings?.map((team) => (
              <tr key={team.team_id} className={team.is_user_team ? 'user-team-row' : ''}>
                <td className="rank-cell">
                  {team.rank === 1 && <span className="medal gold">1</span>}
                  {team.rank === 2 && <span className="medal silver">2</span>}
                  {team.rank === 3 && <span className="medal bronze">3</span>}
                  {team.rank > 3 && <span>{team.rank}</span>}
                </td>
                <td className="team-cell">
                  {team.team_name}
                  {team.is_user_team && <span className="you-badge">YOU</span>}
                </td>
                <td className="points-cell">{team.total_points?.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Category Leaders */}
      {category_leaders && category_leaders.length > 0 && (
        <div className="category-leaders-card">
          <h3>Your Category Leaders</h3>
          <div className="leaders-grid">
            {category_leaders.map((leader) => (
              <div key={leader.category} className="leader-item">
                <span className="leader-category">{leader.category}</span>
                <span className="leader-name">{leader.player_name}</span>
                <span className="leader-value">
                  {formatLeaderValue(leader.category, leader.per_game_value)}
                </span>
                <span className={`leader-z ${leader.z_score >= 0 ? 'positive' : 'negative'}`}>
                  z: {leader.z_score >= 0 ? '+' : ''}{leader.z_score}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Convert number to ordinal (1st, 2nd, 3rd, etc.)
 */
function getOrdinal(n) {
  if (!n) return '-';
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

/**
 * Format leader value based on category type
 */
function formatLeaderValue(category, value) {
  if (typeof value !== 'number') return value ?? '-';

  // Format percentages
  if (category === 'FG%' || category === 'FT%') {
    return (value * 100).toFixed(1) + '%';
  }

  return value.toFixed(1);
}

/**
 * Map category name to stat key in per_game_stats
 */
function mapCategoryToStat(category) {
  const mapping = {
    'FG%': 'fg_pct',
    'FT%': 'ft_pct',
    '3PM': '3pm',
    'PTS': 'pts',
    'REB': 'reb',
    'AST': 'ast',
    'STL': 'stl',
    'BLK': 'blk',
    'TO': 'to',
  };
  return mapping[category] || category.toLowerCase();
}

export default SeasonRecap;

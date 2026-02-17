import React from 'react';
import PropTypes from 'prop-types';

/**
 * Movement arrow icons
 */
const ArrowUp = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 4l-8 8h5v8h6v-8h5z" />
  </svg>
);

const ArrowDown = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 20l8-8h-5V4H9v8H4z" />
  </svg>
);

/**
 * Get CSS class for category POINTS (color coding by tier)
 * Points: 10 = 1st place, 9 = 2nd, ..., 1 = 10th place
 *
 * @param {number} points - Points earned in category (1-10 for 10-team league)
 * @param {number} totalTeams - Number of teams in league
 * @returns {string} CSS class name
 */
const getCategoryPointsClass = (points, totalTeams) => {
  if (typeof points !== 'number' || !totalTeams) return '';

  // For a 10-team league: 10-9 = top tier, 8-6 = middle, 5-1 = bottom
  // Generalize: top 20% of points = excellent, next 30% = good, rest = poor
  const maxPoints = totalTeams;

  if (points >= maxPoints - 1) return 'points-top';      // 10-9 (1st-2nd place) - GREEN
  if (points >= maxPoints - 4) return 'points-middle';   // 8-6 (3rd-5th place) - YELLOW
  return 'points-bottom';                                 // 5-1 (6th-10th place) - RED
};

/**
 * StandingsTable - Reusable component for displaying league standings
 * Supports both H2H (record-based) and Roto (category-based) formats
 */
/**
 * Info icon with tooltip for start limits
 */
const StartLimitsInfoIcon = ({ tooltip }) => (
  <span
    className="start-limits-info-icon"
    title={tooltip}
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      cursor: 'help',
      marginLeft: '8px',
      opacity: 0.7,
    }}
  >
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  </span>
);

function StandingsTable({
  standings = [],
  leagueType = 'H2H',
  showProjected = false,
  showOwner = true,
  showProbability = false,
  probabilityLabel = 'Win %',
  userTeamId = null,
  title = null,
  compact = false,
  onTeamClick = null,
  categories = null, // Custom categories for Roto
  startLimitsEnabled = false, // Show start limits indicator
  irReturns = [], // IR player returns to highlight
}) {
  const isRoto = leagueType.toUpperCase().includes('ROTO');
  const rotoCategories = categories || [];
  const totalTeams = standings.length;

  // Sort standings by rank (or projected rank if showing projected)
  const sortedStandings = [...standings].sort((a, b) => {
    const rankA = showProjected ? (a.projected_rank || a.rank) : a.rank;
    const rankB = showProjected ? (b.projected_rank || b.rank) : b.rank;
    return rankA - rankB;
  });

  /**
   * Get movement value (positive = improved, negative = dropped)
   */
  const getMovement = (team) => {
    if (!showProjected) return 0;
    const currentRank = team.current_rank || team.rank;
    const projectedRank = team.projected_rank || team.rank;
    return currentRank - projectedRank;
  };

  /**
   * Get movement display class
   */
  const getMovementClass = (movement) => {
    if (movement > 0) return 'movement-up';
    if (movement < 0) return 'movement-down';
    return 'movement-none';
  };

  /**
   * Format record display (H2H only)
   */
  const formatRecord = (team) => {
    if (team.record) return team.record;
    if (team.wins !== undefined && team.losses !== undefined) {
      if (team.ties !== undefined && team.ties > 0) {
        return `${team.wins}-${team.losses}-${team.ties}`;
      }
      return `${team.wins}-${team.losses}`;
    }
    return '—';
  };

  /**
   * Format probability as percentage
   */
  const formatProbability = (prob) => {
    if (prob === undefined || prob === null) return '—';
    const percentage = typeof prob === 'number' ? prob * 100 : parseFloat(prob);
    return `${Math.round(percentage)}%`;
  };

  /**
   * Get category POINTS for a team (not rank!)
   * Points: 10 = 1st place, 9 = 2nd, ..., 1 = last place
   *
   * The API returns points in team.categories[stat_key] (primary)
   * or team.category_points[stat_key] (alias)
   */
  const getCategoryPoints = (team, categoryKey) => {
    // Primary: use 'categories' dict which contains POINTS
    const points = team.categories?.[categoryKey]
      ?? team.category_points?.[categoryKey]
      ?? team[categoryKey];

    return typeof points === 'number' ? points : '—';
  };

  /**
   * Get category RANK for tooltip (1 = best, 10 = worst)
   */
  const getCategoryRank = (team, categoryKey) => {
    const rank = team.category_ranks?.[categoryKey];
    return typeof rank === 'number' ? rank : null;
  };

  /**
   * Get total Roto points (sum of all category points)
   * Higher is better
   */
  const getTotalPoints = (team) => {
    if (team.total_points !== undefined) return team.total_points;
    if (team.roto_points !== undefined) return team.roto_points;

    // Calculate from category points if available
    let total = 0;
    let hasData = false;
    rotoCategories.forEach(cat => {
      const points = getCategoryPoints(team, cat.key);
      if (typeof points === 'number') {
        total += points;
        hasData = true;
      }
    });
    return hasData ? total : '—';
  };

  /**
   * Check if team is user's team
   */
  const isUserTeam = (team) => {
    if (userTeamId === null) return false;
    return team.team_id === userTeamId || team.id === userTeamId || team.is_user_team === true;
  };

  /**
   * Handle row click
   */
  const handleRowClick = (team) => {
    if (onTeamClick) {
      onTeamClick(team);
    }
  };

  if (!standings || standings.length === 0) {
    return (
      <div className="standings-table-container">
        {title && <h3 className="standings-title">{title}</h3>}
        <div className="standings-empty">No standings data available</div>
      </div>
    );
  }

  // Handle Roto leagues without categories
  if (isRoto && (!rotoCategories || rotoCategories.length === 0)) {
    return (
      <div className="standings-table-container">
        {title && <h3 className="standings-title">{title}</h3>}
        <div className="standings-empty">
          No category data available. Please refresh league data.
        </div>
      </div>
    );
  }

  // Render Roto CURRENT standings table - showing POINTS in each category
  if (isRoto && !showProjected) {
    return (
      <div className={`standings-table-container roto ${compact ? 'compact' : ''}`}>
        {title && <h3 className="standings-title">{title}</h3>}
        <div className="standings-table-wrapper">
          <table className="standings-table roto-table">
            <thead>
              <tr>
                <th className="col-rank">#</th>
                <th className="col-team">Team</th>
                {rotoCategories.map(cat => (
                  <th key={cat.key} className="col-category" title={cat.label}>
                    {cat.abbr}
                  </th>
                ))}
                <th className="col-total">Total</th>
              </tr>
            </thead>
            <tbody>
              {sortedStandings.map((team, index) => {
                const isUser = isUserTeam(team);
                const rank = team.rank || index + 1;

                return (
                  <tr
                    key={team.team_id || team.id || index}
                    className={`
                      standings-row
                      ${isUser ? 'user-team-row' : ''}
                      ${onTeamClick ? 'clickable' : ''}
                    `}
                    onClick={() => handleRowClick(team)}
                  >
                    <td className="col-rank">
                      <span className="rank-number">{rank}</span>
                    </td>
                    <td className="col-team">
                      <div className="team-info">
                        <span className="team-name">{team.team_name || team.name}</span>
                        {isUser && <span className="user-badge">You</span>}
                      </div>
                    </td>
                    {rotoCategories.map(cat => {
                      // Get POINTS (10 for 1st, 9 for 2nd, etc.) - NOT rank
                      const catPoints = getCategoryPoints(team, cat.key);
                      const catRank = getCategoryRank(team, cat.key);
                      const pointsClass = typeof catPoints === 'number'
                        ? getCategoryPointsClass(catPoints, totalTeams)
                        : '';

                      // Build tooltip showing rank
                      const tooltip = catRank
                        ? `${cat.label}: Rank ${catRank} = ${catPoints} pts`
                        : cat.label;

                      return (
                        <td key={cat.key} className={`col-category ${pointsClass}`}>
                          <span className="category-points" title={tooltip}>
                            {catPoints}
                          </span>
                        </td>
                      );
                    })}
                    <td className="col-total">
                      <span className="total-points">{getTotalPoints(team)}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="roto-legend">
          <span className="legend-item"><span className="dot points-top"></span> 10-9 pts (1st-2nd)</span>
          <span className="legend-item"><span className="dot points-middle"></span> 8-6 pts (3rd-5th)</span>
          <span className="legend-item"><span className="dot points-bottom"></span> 5-1 pts (6th-10th)</span>
        </div>
      </div>
    );
  }

  // Render Roto PROJECTED standings table - showing POINTS in each category
  if (isRoto && showProjected) {
    /**
     * Get projected category POINTS for a team (not rank!)
     * Points: 10 = 1st place, 9 = 2nd, ..., 1 = last place
     */
    const getProjectedCategoryPoints = (team, categoryKey) => {
      // Primary: use 'categories' dict which contains POINTS
      const points = team.categories?.[categoryKey]
        ?? team.projected_category_points?.[categoryKey]
        ?? team.category_points?.[categoryKey];
      return typeof points === 'number' ? points : '—';
    };

    /**
     * Get projected category RANK for tooltip
     */
    const getProjectedCategoryRank = (team, categoryKey) => {
      const rank = team.projected_category_ranks?.[categoryKey]
        ?? team.category_ranks?.[categoryKey];
      return typeof rank === 'number' ? rank : null;
    };

    /**
     * Get projected total points
     */
    const getProjectedTotalPoints = (team) => {
      if (team.projected_total_points !== undefined) return team.projected_total_points;
      if (team.projected_roto_points !== undefined) return team.projected_roto_points;
      if (team.roto_points !== undefined) return team.roto_points;
      if (team.total_points !== undefined) return team.total_points;
      return '—';
    };

    // Build IR return note if applicable
    const irReturnNote = irReturns.length > 0 && irReturns[0]?.name
      ? `Projections assume ${irReturns[0].name} returns ${
          irReturns[0].projected_return_date
            ? new Date(irReturns[0].projected_return_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
            : 'soon'
        }${irReturns[0].replacing_player ? `, replacing ${irReturns[0].replacing_player}` : ''}`
      : null;

    return (
      <div className={`standings-table-container roto roto-projected ${compact ? 'compact' : ''}`}>
        {title && (
          <h3 className="standings-title">
            {title}
            {startLimitsEnabled && (
              <StartLimitsInfoIcon
                tooltip="Projections are adjusted for position start limits using day-by-day simulation. Each position has a maximum number of games that can be started."
              />
            )}
          </h3>
        )}
        {irReturnNote && (
          <div className="ir-return-note" style={{
            fontSize: '0.8rem',
            color: '#10b981',
            marginBottom: '8px',
            padding: '6px 10px',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
              <path d="M22 4L12 14.01l-3-3" />
            </svg>
            {irReturnNote}
          </div>
        )}
        <div className="standings-table-wrapper">
          <table className="standings-table roto-table roto-projected-table">
            <thead>
              <tr>
                <th className="col-rank">#</th>
                <th className="col-team">Team</th>
                {rotoCategories.map(cat => (
                  <th key={cat.key} className="col-category" title={cat.label}>
                    {cat.abbr}
                  </th>
                ))}
                <th className="col-total">Total</th>
                <th className="col-movement">Move</th>
              </tr>
            </thead>
            <tbody>
              {sortedStandings.map((team, index) => {
                const isUser = isUserTeam(team);
                const projectedRank = team.projected_rank || team.projected_standing || index + 1;
                const currentRank = team.current_rank || team.current_standing || team.rank || projectedRank;
                const movement = currentRank - projectedRank;

                // Count categories with high points (winning) and low points (losing)
                let catsWinning = 0;  // 10-9 pts (top tier)
                let catsLosing = 0;   // 5-1 pts (bottom tier)
                rotoCategories.forEach(cat => {
                  const catPoints = getProjectedCategoryPoints(team, cat.key);
                  if (typeof catPoints === 'number') {
                    if (catPoints >= totalTeams - 1) catsWinning++;  // 10-9 pts
                    if (catPoints <= 5) catsLosing++;                 // 5-1 pts
                  }
                });

                return (
                  <tr
                    key={team.team_id || team.id || index}
                    className={`
                      standings-row
                      ${isUser ? 'user-team-row' : ''}
                      ${onTeamClick ? 'clickable' : ''}
                    `}
                    onClick={() => handleRowClick(team)}
                  >
                    <td className="col-rank">
                      <span className="rank-number">{projectedRank}</span>
                    </td>
                    <td className="col-team">
                      <div className="team-info">
                        <span className="team-name">{team.team_name || team.name}</span>
                        {isUser && <span className="user-badge">You</span>}
                        <span className="category-summary">
                          <span className="cats-won" title="Categories with 10-9 pts">{catsWinning}W</span>
                          <span className="cats-lost" title="Categories with 5-1 pts">{catsLosing}L</span>
                        </span>
                      </div>
                    </td>
                    {rotoCategories.map(cat => {
                      // Get POINTS (10 for 1st, 9 for 2nd, etc.) - NOT rank
                      const catPoints = getProjectedCategoryPoints(team, cat.key);
                      const catRank = getProjectedCategoryRank(team, cat.key);
                      const pointsClass = typeof catPoints === 'number'
                        ? getCategoryPointsClass(catPoints, totalTeams)
                        : '';

                      // Build tooltip showing rank
                      const tooltip = catRank
                        ? `${cat.label}: Rank ${catRank} = ${catPoints} pts`
                        : cat.label;

                      return (
                        <td key={cat.key} className={`col-category ${pointsClass}`}>
                          <span className="category-points" title={tooltip}>
                            {catPoints}
                          </span>
                        </td>
                      );
                    })}
                    <td className="col-total">
                      <span className="total-points">{getProjectedTotalPoints(team)}</span>
                    </td>
                    <td className="col-movement">
                      <span className={`movement-indicator ${getMovementClass(movement)}`}>
                        {movement > 0 && (
                          <>
                            <ArrowUp />
                            <span className="movement-value">+{movement}</span>
                          </>
                        )}
                        {movement < 0 && (
                          <>
                            <ArrowDown />
                            <span className="movement-value">{movement}</span>
                          </>
                        )}
                        {movement === 0 && <span className="movement-dash">—</span>}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="roto-legend projected-legend">
          <span className="legend-item"><span className="dot points-top"></span> 10-9 pts (1st-2nd)</span>
          <span className="legend-item"><span className="dot points-middle"></span> 8-6 pts (3rd-5th)</span>
          <span className="legend-item"><span className="dot points-bottom"></span> 5-1 pts (6th-10th)</span>
        </div>
      </div>
    );
  }

  // Render H2H standings table
  return (
    <div className={`standings-table-container ${compact ? 'compact' : ''}`}>
      {title && <h3 className="standings-title">{title}</h3>}
      <div className="standings-table-wrapper">
        <table className="standings-table">
          <thead>
            <tr>
              <th className="col-rank">#</th>
              <th className="col-team">Team</th>
              {showOwner && <th className="col-owner">Owner</th>}
              {!isRoto && <th className="col-record">Record</th>}
              {isRoto && !showProjected && <th className="col-total">Points</th>}
              {showProbability && <th className="col-probability">{probabilityLabel}</th>}
              {showProjected && <th className="col-movement">Move</th>}
            </tr>
          </thead>
          <tbody>
            {sortedStandings.map((team, index) => {
              const movement = getMovement(team);
              const isUser = isUserTeam(team);
              const rank = showProjected ? (team.projected_rank || team.rank) : team.rank;

              return (
                <tr
                  key={team.team_id || team.id || index}
                  className={`
                    standings-row
                    ${isUser ? 'user-team-row' : ''}
                    ${onTeamClick ? 'clickable' : ''}
                  `}
                  onClick={() => handleRowClick(team)}
                >
                  <td className="col-rank">
                    <span className="rank-number">{rank || index + 1}</span>
                  </td>
                  <td className="col-team">
                    <div className="team-info">
                      <span className="team-name">{team.team_name || team.name}</span>
                      {isUser && <span className="user-badge">You</span>}
                    </div>
                  </td>
                  {showOwner && (
                    <td className="col-owner">
                      <span className="owner-name">{team.owner_name || team.owner || '—'}</span>
                    </td>
                  )}
                  {!isRoto && (
                    <td className="col-record">
                      <span className="record">{formatRecord(team)}</span>
                    </td>
                  )}
                  {isRoto && !showProjected && (
                    <td className="col-total">
                      <span className="total-points">{getTotalPoints(team)}</span>
                    </td>
                  )}
                  {showProbability && (
                    <td className="col-probability">
                      <div className="probability-cell">
                        <div className="probability-bar-bg">
                          <div
                            className="probability-bar-fill"
                            style={{ width: `${(team.win_probability || 0) * 100}%` }}
                          />
                        </div>
                        <span className="probability-text">
                          {formatProbability(team.win_probability)}
                        </span>
                      </div>
                    </td>
                  )}
                  {showProjected && (
                    <td className="col-movement">
                      <span className={`movement-indicator ${getMovementClass(movement)}`}>
                        {movement > 0 && (
                          <>
                            <ArrowUp />
                            <span className="movement-value">{movement}</span>
                          </>
                        )}
                        {movement < 0 && (
                          <>
                            <ArrowDown />
                            <span className="movement-value">{Math.abs(movement)}</span>
                          </>
                        )}
                        {movement === 0 && <span className="movement-dash">—</span>}
                      </span>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

StandingsTable.propTypes = {
  standings: PropTypes.arrayOf(
    PropTypes.shape({
      team_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      team_name: PropTypes.string,
      name: PropTypes.string,
      owner_name: PropTypes.string,
      owner: PropTypes.string,
      rank: PropTypes.number,
      current_rank: PropTypes.number,
      projected_rank: PropTypes.number,
      record: PropTypes.string,
      wins: PropTypes.number,
      losses: PropTypes.number,
      ties: PropTypes.number,
      win_probability: PropTypes.number,
      is_user_team: PropTypes.bool,
      category_ranks: PropTypes.object,
      categories: PropTypes.object,
      total_points: PropTypes.number,
      roto_points: PropTypes.number,
    })
  ),
  leagueType: PropTypes.string,
  showProjected: PropTypes.bool,
  showOwner: PropTypes.bool,
  showProbability: PropTypes.bool,
  probabilityLabel: PropTypes.string,
  userTeamId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  title: PropTypes.string,
  compact: PropTypes.bool,
  onTeamClick: PropTypes.func,
  categories: PropTypes.arrayOf(
    PropTypes.shape({
      key: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
      abbr: PropTypes.string.isRequired,
    })
  ),
  startLimitsEnabled: PropTypes.bool,
  irReturns: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string,
      projected_return_date: PropTypes.string,
      replacing_player: PropTypes.string,
    })
  ),
};

export default StandingsTable;

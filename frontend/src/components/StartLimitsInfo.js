import React from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '../context/ThemeContext';

/**
 * Info icon with tooltip
 */
const InfoIcon = ({ tooltip, isDark }) => (
  <span
    className="info-icon-wrapper"
    title={tooltip}
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      cursor: 'help',
      marginLeft: '6px',
    }}
  >
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke={isDark ? '#94a3b8' : '#64748b'}
      strokeWidth="2"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  </span>
);

/**
 * Get color class based on usage percentage
 * Color coding: <60% = green, 60-85% = yellow, >85% = red
 */
const getUsageColorClass = (used, limit) => {
  if (!limit || limit === 0) return '';
  const pct = (used / limit) * 100;
  if (pct > 85) return 'usage-critical';      // Red - over 85%
  if (pct >= 60) return 'usage-moderate';     // Yellow - 60-85%
  return 'usage-good';                         // Green - under 60%
};

/**
 * Format position name for display
 */
const formatPositionName = (posName) => {
  const shortNames = {
    'UTIL': 'UTIL',
    'G': 'G',
    'F': 'F',
    'SG/SF': 'SG/SF',
    'G/F': 'G/F',
    'PF/C': 'PF/C',
    'F/C': 'F/C',
  };
  return shortNames[posName] || posName;
};

/**
 * StartLimitsInfo - Display position start limits and IR return info
 */
function StartLimitsInfo({
  startLimits = {},
  showIRReturns = true,
  compact = false,
  title = 'Position Starts',
}) {
  const { isDark } = useTheme();

  if (!startLimits?.enabled) {
    return null;
  }

  const positionLimits = startLimits.position_limits || {};
  const irPlayers = startLimits.ir_players || [];
  const description = startLimits.description || '';

  // Sort positions by slot_id for consistent ordering
  const sortedPositions = Object.entries(positionLimits)
    .sort((a, b) => (a[1].slot_id || 0) - (b[1].slot_id || 0));

  return (
    <div className={`start-limits-info ${compact ? 'compact' : ''} ${isDark ? 'dark' : ''}`}>
      {/* Header */}
      <div className="start-limits-header">
        <h4 className="start-limits-title">
          {title}
          <InfoIcon
            tooltip={description || 'Projections account for position start limits using day-by-day simulation.'}
            isDark={isDark}
          />
        </h4>
      </div>

      {/* Position Grid */}
      <div className="position-grid">
        {sortedPositions.map(([posName, data]) => {
          // Read CURRENT starts from ESPN (games_used), NOT end-of-season projected
          // Backend sends 'games_used' for current ESPN starts
          const used = data.games_used || data.starts_used || 0;
          const limit = data.games_limit || 82;
          const remaining = limit - used;
          const usagePct = limit > 0 ? (used / limit) * 100 : 0;
          const colorClass = getUsageColorClass(used, limit);

          return (
            <div
              key={posName}
              className={`position-item ${colorClass}`}
              title={`${posName}: ${used}/${limit} used (${remaining} remaining)`}
            >
              <div className="position-name">{formatPositionName(posName)}</div>
              <div className="position-stats">
                {/* Display format: "54/82 used (66%)" */}
                <span className="used-value">{used}</span>
                <span className="stats-separator">/</span>
                <span className="limit-value">{limit}</span>
                <span className="usage-pct">({Math.round(usagePct)}%)</span>
              </div>
              <div className="usage-bar-container">
                <div
                  className={`usage-bar-fill ${colorClass}`}
                  style={{ width: `${Math.min(100, usagePct)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* IR Returns Section */}
      {showIRReturns && irPlayers.length > 0 && (
        <div className="ir-returns-section">
          <h5 className="ir-returns-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
              <path d="M22 4L12 14.01l-3-3" />
            </svg>
            Projected IR Returns
          </h5>
          {irPlayers.map((player, idx) => (
            <div key={player.player_id || idx} className="ir-player-item">
              <div className="ir-player-info">
                <span className="ir-player-name">{player.name}</span>
                {player.projected_return_date && (
                  <span className="ir-return-date">
                    Returns: {new Date(player.projected_return_date).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric'
                    })}
                  </span>
                )}
              </div>
              {player.replacing_player && (
                <div className="ir-replacement-info">
                  <span className="replacement-label">Replaces:</span>
                  <span className="replacement-name">{player.replacing_player}</span>
                </div>
              )}
              {player.games_after_return > 0 && (
                <div className="ir-games-info">
                  <span className="games-label">+{player.games_after_return} games</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Legend */}
      {!compact && (
        <div className="start-limits-legend">
          <span className="legend-item">
            <span className="legend-dot usage-good"></span>
            <span className="legend-text">&lt;60%</span>
          </span>
          <span className="legend-item">
            <span className="legend-dot usage-moderate"></span>
            <span className="legend-text">60-85%</span>
          </span>
          <span className="legend-item">
            <span className="legend-dot usage-critical"></span>
            <span className="legend-text">&gt;85%</span>
          </span>
        </div>
      )}
    </div>
  );
}

StartLimitsInfo.propTypes = {
  startLimits: PropTypes.shape({
    enabled: PropTypes.bool,
    description: PropTypes.string,
    position_limits: PropTypes.objectOf(
      PropTypes.shape({
        slot_id: PropTypes.number,
        slots: PropTypes.number,
        games_limit: PropTypes.number,
        games_used: PropTypes.number,      // Current ESPN starts (preferred)
        starts_used: PropTypes.number,     // Alias for backwards compatibility
        games_remaining: PropTypes.number,
      })
    ),
    ir_players: PropTypes.arrayOf(
      PropTypes.shape({
        player_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
        name: PropTypes.string,
        projected_return_date: PropTypes.string,
        replacing_player: PropTypes.string,
        games_after_return: PropTypes.number,
        will_return: PropTypes.bool,
      })
    ),
    starting_players: PropTypes.number,
    benched_players: PropTypes.number,
  }),
  showIRReturns: PropTypes.bool,
  compact: PropTypes.bool,
  title: PropTypes.string,
};

export default StartLimitsInfo;

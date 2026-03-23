import React, { useState, useEffect, useCallback } from 'react';

/**
 * DailyLineup - Shows optimized daily lineup with position assignments
 *
 * Displays start/sit decisions for each day with date navigation,
 * showing which players should start based on z-score optimization.
 */
function DailyLineup({ leagueId }) {
  const [date, setDate] = useState(new Date());
  const [lineup, setLineup] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /**
   * Format date as YYYY-MM-DD for API calls
   */
  const formatDateForApi = (d) => {
    return d.toISOString().split('T')[0];
  };

  /**
   * Fetch lineup data for selected date
   */
  const loadLineup = useCallback(async (selectedDate) => {
    try {
      setLoading(true);
      setError(null);

      const dateStr = formatDateForApi(selectedDate);
      console.log('Fetching daily lineup for:', dateStr);

      const response = await fetch(
        `/api/leagues/${leagueId}/daily-lineup?date=${dateStr}`
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        if (response.status === 404) {
          setError(`No lineup available for ${dateStr}. ${errorData.message || ''}`);
          setLineup(null);
        } else {
          throw new Error(errorData.error || `HTTP ${response.status}`);
        }
        return;
      }

      const data = await response.json();
      console.log('Lineup loaded:', data);
      setLineup(data);

    } catch (err) {
      console.error('Error loading lineup:', err);
      setError(err.message);
      setLineup(null);
    } finally {
      setLoading(false);
    }
  }, [leagueId]);

  /**
   * Load lineup when date changes
   */
  useEffect(() => {
    loadLineup(date);
  }, [date, loadLineup]);

  /**
   * Navigate to previous day
   */
  const goToPreviousDay = () => {
    const prevDay = new Date(date);
    prevDay.setDate(prevDay.getDate() - 1);
    setDate(prevDay);
  };

  /**
   * Navigate to next day
   */
  const goToNextDay = () => {
    const nextDay = new Date(date);
    nextDay.setDate(nextDay.getDate() + 1);
    setDate(nextDay);
  };

  /**
   * Go to today
   */
  const goToToday = () => {
    setDate(new Date());
  };

  /**
   * Get color class for position chip
   */
  const getPositionColorClass = (position) => {
    const colors = {
      'PG': 'position-pg',
      'SG': 'position-sg',
      'SF': 'position-sf',
      'PF': 'position-pf',
      'C': 'position-c',
      'G': 'position-g',
      'F': 'position-f',
      'UTIL': 'position-util'
    };
    // Handle positions like UTIL_2
    const basePos = position.split('_')[0];
    return colors[basePos] || 'position-default';
  };

  /**
   * Get ordered positions for display
   */
  const getOrderedPositions = () => {
    return ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'];
  };

  /**
   * Sort lineup slots by position order
   */
  const getSortedLineupSlots = () => {
    if (!lineup?.lineup_slots) return [];

    const positionOrder = getOrderedPositions();
    const slots = Object.entries(lineup.lineup_slots);

    return slots.sort((a, b) => {
      const posA = a[0].split('_')[0];
      const posB = b[0].split('_')[0];
      const idxA = positionOrder.indexOf(posA);
      const idxB = positionOrder.indexOf(posB);

      if (idxA !== idxB) return idxA - idxB;
      // Same base position, sort by suffix (UTIL_1 before UTIL_2)
      return a[0].localeCompare(b[0]);
    });
  };

  // Render loading state
  if (loading && !lineup) {
    return (
      <div className="daily-lineup-container">
        <div className="daily-lineup-header">
          <h2>Daily Lineup</h2>
        </div>
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading lineup...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="daily-lineup-container">
      {/* Header with date navigation */}
      <div className="daily-lineup-header">
        <h2>Daily Lineup</h2>
        <div className="date-navigation">
          <button
            className="nav-btn"
            onClick={goToPreviousDay}
            title="Previous day"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M15 18l-6-6 6-6" />
            </svg>
          </button>

          <button
            className="btn btn-secondary today-btn"
            onClick={goToToday}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
              <line x1="16" y1="2" x2="16" y2="6" />
              <line x1="8" y1="2" x2="8" y2="6" />
              <line x1="3" y1="10" x2="21" y2="10" />
            </svg>
            Today
          </button>

          <button
            className="nav-btn"
            onClick={goToNextDay}
            title="Next day"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 18l6-6-6-6" />
            </svg>
          </button>
        </div>
        <div className="date-display">
          <span className="date-text">{lineup?.formatted_date || date.toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'long',
            day: 'numeric',
            year: 'numeric'
          })}</span>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="alert alert-warning">
          {error}
          <button className="btn btn-sm btn-secondary" onClick={() => loadLineup(date)} style={{ marginLeft: '10px' }}>
            Retry
          </button>
        </div>
      )}

      {/* Loading overlay when refreshing */}
      {loading && lineup && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
        </div>
      )}

      {/* Lineup content */}
      {lineup && (
        <>
          {/* Summary stats */}
          <div className="lineup-summary">
            <div className="summary-stat">
              <span className="stat-label">Games Today</span>
              <span className="stat-value">{lineup.summary?.players_with_games || 0}</span>
            </div>
            <div className="summary-stat">
              <span className="stat-label">Starters</span>
              <span className="stat-value">{lineup.summary?.starters || 0}</span>
            </div>
            <div className="summary-stat">
              <span className="stat-label">Benched</span>
              <span className="stat-value">{lineup.summary?.benched || 0}</span>
            </div>
            {lineup.summary?.slots_at_limit?.length > 0 && (
              <div className="summary-stat warning">
                <span className="stat-label">At Limit</span>
                <span className="stat-value">{lineup.summary.slots_at_limit.join(', ')}</span>
              </div>
            )}
          </div>

          {/* Starting Lineup */}
          <div className="lineup-section">
            <h3 className="section-title">
              <span className="title-icon start">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                </svg>
              </span>
              Starting Lineup
            </h3>
            <div className="lineup-list">
              {getSortedLineupSlots().map(([slot, data]) => (
                <div key={slot} className="lineup-row starter">
                  <span className={`position-chip ${getPositionColorClass(slot)}`}>
                    {slot.includes('_') ? slot.split('_')[0] : slot}
                  </span>
                  <div className="player-info">
                    <span className="player-name">{data.player?.name || 'Unknown'}</span>
                    <span className="player-team">{data.player?.team || ''}</span>
                  </div>
                  <div className="game-info">
                    <span className="has-game">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                      Playing
                    </span>
                  </div>
                </div>
              ))}
              {getSortedLineupSlots().length === 0 && (
                <div className="empty-message">No starters assigned</div>
              )}
            </div>
          </div>

          {/* Bench (players with games but not starting) */}
          {lineup.bench && lineup.bench.length > 0 && (
            <div className="lineup-section bench-section">
              <h3 className="section-title">
                <span className="title-icon bench">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="2" y="6" width="20" height="12" rx="2" />
                    <line x1="6" y1="10" x2="6" y2="14" />
                    <line x1="10" y1="10" x2="10" y2="14" />
                    <line x1="14" y1="10" x2="14" y2="14" />
                    <line x1="18" y1="10" x2="18" y2="14" />
                  </svg>
                </span>
                Bench (Have Game)
              </h3>
              <div className="lineup-list bench-list">
                {lineup.bench.map((player, idx) => (
                  <div key={player.player?.id || idx} className="lineup-row bench">
                    <span className="position-chip position-bench">BN</span>
                    <div className="player-info">
                      <span className="player-name">{player.player?.name || 'Unknown'}</span>
                      <span className="player-team">{player.player?.team || ''}</span>
                    </div>
                    <div className="game-info">
                      <span className="has-game">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                        Playing
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Injured (players who are OUT but their team has a game) */}
          {lineup.injured && lineup.injured.length > 0 && (
            <div className="lineup-section injured-section">
              <h3 className="section-title">
                <span className="title-icon injured">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
                    <path d="M12 8v4" />
                    <path d="M12 16h.01" />
                  </svg>
                </span>
                Injured (OUT)
              </h3>
              <div className="lineup-list injured-list">
                {lineup.injured.map((player, idx) => (
                  <div key={player.player?.id || idx} className="lineup-row injured">
                    <span className="position-chip position-injured">OUT</span>
                    <div className="player-info">
                      <span className="player-name">{player.player?.name || 'Unknown'}</span>
                      <span className="player-team">{player.player?.team || ''}</span>
                    </div>
                    <div className="game-info">
                      <span className="injured-status">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="10" />
                          <line x1="15" y1="9" x2="9" y2="15" />
                          <line x1="9" y1="9" x2="15" y2="15" />
                        </svg>
                        Out
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No Game (players without games today) */}
          {lineup.no_game && lineup.no_game.length > 0 && (
            <div className="lineup-section no-game-section">
              <h3 className="section-title">
                <span className="title-icon no-game">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
                  </svg>
                </span>
                No Game Today
              </h3>
              <div className="lineup-list no-game-list">
                {lineup.no_game.map((player, idx) => (
                  <div key={player.player?.id || idx} className="lineup-row no-game">
                    <span className="position-chip position-no-game">OFF</span>
                    <div className="player-info">
                      <span className="player-name">{player.player?.name || 'Unknown'}</span>
                      <span className="player-team">{player.player?.team || ''}</span>
                    </div>
                    <div className="game-info">
                      <span className="no-game">No game</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* IR */}
          {lineup.ir && lineup.ir.length > 0 && (
            <div className="lineup-section ir-section">
              <h3 className="section-title">
                <span className="title-icon ir">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                  </svg>
                </span>
                Injured Reserve
              </h3>
              <div className="lineup-list ir-list">
                {lineup.ir.map((player, idx) => (
                  <div key={player.player?.id || idx} className="lineup-row ir">
                    <span className="position-chip position-ir">IR</span>
                    <div className="player-info">
                      <span className="player-name">{player.player?.name || 'Unknown'}</span>
                      <span className="player-injury">
                        {player.injury_status}
                        {player.return_date && ` (Return: ${player.return_date})`}
                      </span>
                    </div>
                    <div className="game-info">
                      <span className="no-game">Injured</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default DailyLineup;

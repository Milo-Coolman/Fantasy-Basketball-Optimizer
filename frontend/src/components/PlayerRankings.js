import React, { useState, useEffect, useCallback } from 'react';

/**
 * PlayerRankings - Universal player rankings based on z-scores
 *
 * Shows all NBA players ranked by total z-score for the league's categories.
 * Supports sorting by any column, filtering by position, and search.
 */
function PlayerRankings({ leagueId }) {
  const [players, setPlayers] = useState([]);
  const [filteredPlayers, setFilteredPlayers] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [cachedAt, setCachedAt] = useState(null);

  // Filters
  const [searchTerm, setSearchTerm] = useState('');
  const [positionFilter, setPositionFilter] = useState('ALL');
  const [rosterFilter, setRosterFilter] = useState('ALL'); // ALL, ROSTERED, FREE_AGENT

  // Sorting
  const [sortBy, setSortBy] = useState('total_z_score');
  const [sortDirection, setSortDirection] = useState('desc');

  // Player detail modal
  const [selectedPlayer, setSelectedPlayer] = useState(null);

  // Pagination
  const [displayCount, setDisplayCount] = useState(50);

  /**
   * Fetch player rankings from backend
   */
  const loadRankings = useCallback(async (forceRefresh = false) => {
    try {
      setLoading(true);
      setError(null);

      console.log('Fetching player rankings for league:', leagueId);
      const url = `/api/leagues/${leagueId}/player-rankings${forceRefresh ? '?force_refresh=true' : ''}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Player rankings loaded:', data.total_players, 'players');

      setPlayers(data.players || []);
      setCategories(data.categories || []);
      setCachedAt(data.cached_at);

    } catch (err) {
      console.error('Failed to load player rankings:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [leagueId]);

  /**
   * Load rankings on mount
   */
  useEffect(() => {
    loadRankings();
  }, [loadRankings]);

  /**
   * Apply filters and sorting when data or filters change
   */
  useEffect(() => {
    let filtered = [...players];

    // Search filter
    if (searchTerm) {
      const search = searchTerm.toLowerCase();
      filtered = filtered.filter(p =>
        p.name.toLowerCase().includes(search) ||
        p.team.toLowerCase().includes(search)
      );
    }

    // Position filter
    if (positionFilter !== 'ALL') {
      filtered = filtered.filter(p =>
        p.positions && p.positions.includes(positionFilter)
      );
    }

    // Roster status filter
    if (rosterFilter === 'ROSTERED') {
      filtered = filtered.filter(p => p.is_rostered);
    } else if (rosterFilter === 'FREE_AGENT') {
      filtered = filtered.filter(p => !p.is_rostered);
    }

    // Sort
    filtered.sort((a, b) => {
      let aValue, bValue;

      if (sortBy === 'total_z_score') {
        aValue = a.total_z_score || 0;
        bValue = b.total_z_score || 0;
      } else if (sortBy === 'name') {
        return sortDirection === 'desc'
          ? b.name.localeCompare(a.name)
          : a.name.localeCompare(b.name);
      } else if (sortBy === 'games_played') {
        aValue = a.games_played || 0;
        bValue = b.games_played || 0;
      } else {
        // Sorting by category z-score
        aValue = a.category_z_scores?.[sortBy] || 0;
        bValue = b.category_z_scores?.[sortBy] || 0;
      }

      return sortDirection === 'desc' ? bValue - aValue : aValue - bValue;
    });

    setFilteredPlayers(filtered);
  }, [players, searchTerm, positionFilter, rosterFilter, sortBy, sortDirection]);

  /**
   * Handle column sort click
   */
  const handleSort = (column) => {
    if (sortBy === column) {
      // Toggle direction
      setSortDirection(sortDirection === 'desc' ? 'asc' : 'desc');
    } else {
      // New column, default to desc
      setSortBy(column);
      setSortDirection('desc');
    }
  };

  /**
   * Get CSS class for z-score coloring
   */
  const getZScoreClass = (zScore) => {
    if (zScore > 1) return 'z-excellent';
    if (zScore > 0) return 'z-above-avg';
    if (zScore > -1) return 'z-below-avg';
    return 'z-poor';
  };

  /**
   * Format z-score for display
   */
  const formatZScore = (zScore) => {
    if (zScore === undefined || zScore === null) return '0.00';
    const formatted = zScore.toFixed(2);
    return zScore > 0 ? `+${formatted}` : formatted;
  };

  /**
   * Get sort indicator
   */
  const getSortIndicator = (column) => {
    if (sortBy !== column) return null;
    return sortDirection === 'desc' ? ' ▼' : ' ▲';
  };

  // Loading state
  if (loading) {
    return (
      <div className="player-rankings-panel">
        <div className="panel-header">
          <h2>📊 Player Rankings</h2>
        </div>
        <div className="panel-content">
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading player rankings...</p>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="player-rankings-panel">
        <div className="panel-header">
          <h2>📊 Player Rankings</h2>
        </div>
        <div className="panel-content">
          <div className="error-container">
            <p className="error-message">Error loading player rankings</p>
            <p className="error-details">{error}</p>
            <button className="btn btn-primary" onClick={() => loadRankings()}>
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="player-rankings-panel">
      <div className="panel-header">
        <h2>📊 Player Rankings</h2>
        <div className="header-actions">
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => loadRankings(true)}
            title="Refresh rankings (bypass cache)"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      <div className="panel-content">
        {/* Filters Row */}
        <div className="rankings-filters">
          <div className="filter-group">
            <input
              type="text"
              className="search-input"
              placeholder="Search player or team..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <select
              className="filter-select"
              value={positionFilter}
              onChange={(e) => setPositionFilter(e.target.value)}
            >
              <option value="ALL">All Positions</option>
              <option value="PG">PG</option>
              <option value="SG">SG</option>
              <option value="SF">SF</option>
              <option value="PF">PF</option>
              <option value="C">C</option>
            </select>
          </div>

          <div className="filter-group">
            <select
              className="filter-select"
              value={rosterFilter}
              onChange={(e) => setRosterFilter(e.target.value)}
            >
              <option value="ALL">All Players</option>
              <option value="ROSTERED">Rostered</option>
              <option value="FREE_AGENT">Free Agents</option>
            </select>
          </div>
        </div>

        {/* Results summary */}
        <div className="rankings-summary">
          <span className="summary-count">
            Showing {Math.min(displayCount, filteredPlayers.length)} of {filteredPlayers.length} players
          </span>
          {cachedAt && (
            <span className="summary-cache">
              Data from: {new Date(cachedAt).toLocaleString()}
            </span>
          )}
        </div>

        {/* Rankings Table */}
        <div className="rankings-table-container">
          <table className="rankings-table">
            <thead>
              <tr>
                <th className="col-rank">#</th>
                <th
                  className="col-player sortable"
                  onClick={() => handleSort('name')}
                >
                  Player{getSortIndicator('name')}
                </th>
                <th className="col-team">Team</th>
                <th className="col-pos">Pos</th>
                <th className="col-gp">GP</th>
                <th
                  className="col-total-z sortable"
                  onClick={() => handleSort('total_z_score')}
                >
                  Total Z{getSortIndicator('total_z_score')}
                </th>
                {categories.map(cat => (
                  <th
                    key={cat}
                    className="col-category sortable"
                    onClick={() => handleSort(cat)}
                  >
                    {cat}{getSortIndicator(cat)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredPlayers.slice(0, displayCount).map((player, idx) => (
                <tr
                  key={player.player_id}
                  className={`player-row ${player.is_rostered ? 'rostered' : 'free-agent'}`}
                  onClick={() => setSelectedPlayer(player)}
                >
                  <td className="col-rank">{idx + 1}</td>
                  <td className="col-player">
                    <div className="player-name-cell">
                      <span className="player-name">{player.name}</span>
                      {player.injury_status && player.injury_status !== 'ACTIVE' && (
                        <span className="injury-badge">{player.injury_status}</span>
                      )}
                      {!player.is_rostered && (
                        <span className="fa-badge" title="Free Agent">FA</span>
                      )}
                    </div>
                  </td>
                  <td className="col-team">{player.team}</td>
                  <td className="col-pos">{player.positions?.slice(0, 2).join('/')}</td>
                  <td className="col-gp">{player.games_played}</td>
                  <td className={`col-total-z ${getZScoreClass(player.total_z_score)}`}>
                    <strong>{formatZScore(player.total_z_score)}</strong>
                  </td>
                  {categories.map(cat => (
                    <td
                      key={cat}
                      className={`col-category ${getZScoreClass(player.category_z_scores?.[cat] || 0)}`}
                    >
                      {formatZScore(player.category_z_scores?.[cat])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Load More */}
        {filteredPlayers.length > displayCount && (
          <div className="load-more-container">
            <button
              className="btn btn-secondary"
              onClick={() => setDisplayCount(displayCount + 50)}
            >
              Load More ({filteredPlayers.length - displayCount} remaining)
            </button>
          </div>
        )}
      </div>

      {/* Player Detail Modal */}
      {selectedPlayer && (
        <div className="modal-overlay" onClick={() => setSelectedPlayer(null)}>
          <div className="modal-content player-detail-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                {selectedPlayer.name}
                {selectedPlayer.injury_status && selectedPlayer.injury_status !== 'ACTIVE' && (
                  <span className="injury-badge">{selectedPlayer.injury_status}</span>
                )}
              </h3>
              <button className="modal-close" onClick={() => setSelectedPlayer(null)}>×</button>
            </div>
            <div className="modal-body">
              <div className="player-meta">
                <span className="meta-item">{selectedPlayer.team}</span>
                <span className="meta-item">{selectedPlayer.positions?.join(', ')}</span>
                <span className="meta-item">GP: {selectedPlayer.games_played}</span>
                {!selectedPlayer.is_rostered && (
                  <span className="meta-item fa-indicator">Free Agent</span>
                )}
              </div>

              <div className="total-z-display">
                <span className="label">Total Z-Score</span>
                <span className={`value ${getZScoreClass(selectedPlayer.total_z_score)}`}>
                  {formatZScore(selectedPlayer.total_z_score)}
                </span>
              </div>

              <div className="category-breakdown">
                <h4>Category Z-Scores</h4>
                <div className="category-grid">
                  {categories.map(cat => {
                    const zScore = selectedPlayer.category_z_scores?.[cat] || 0;
                    return (
                      <div key={cat} className="category-item">
                        <span className="cat-label">{cat}</span>
                        <span className={`cat-value ${getZScoreClass(zScore)}`}>
                          {formatZScore(zScore)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {selectedPlayer.per_game_stats && (
                <div className="per-game-stats">
                  <h4>Per Game Stats</h4>
                  <div className="stats-grid">
                    {Object.entries(selectedPlayer.per_game_stats).map(([key, value]) => (
                      <div key={key} className="stat-item">
                        <span className="stat-label">{key.toUpperCase()}</span>
                        <span className="stat-value">
                          {typeof value === 'number' ? value.toFixed(1) : value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PlayerRankings;

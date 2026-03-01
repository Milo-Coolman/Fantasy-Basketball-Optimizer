import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { analyzeTrade, fetchTradeSettings, updateTradeSettings, analyzeWaiver } from '../services/api';

/**
 * Icon components
 */
const WaiverIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="16" />
    <line x1="8" y1="12" x2="16" y2="12" />
  </svg>
);

const TradeIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M17 1l4 4-4 4" />
    <path d="M3 11V9a4 4 0 014-4h14" />
    <path d="M7 23l-4-4 4-4" />
    <path d="M21 13v2a4 4 0 01-4 4H3" />
  </svg>
);

const CategoryIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
);

const TrendUpIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
    <polyline points="17 6 23 6 23 12" />
  </svg>
);

const TrendDownIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 18 13.5 8.5 8.5 13.5 1 6" />
    <polyline points="17 18 23 18 23 12" />
  </svg>
);

const ArrowUpIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 4l-8 8h5v8h6v-8h5z" />
  </svg>
);

const ArrowDownIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 20l8-8h-5V4H9v8H4z" />
  </svg>
);

const RotoIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 6v6l4 2" />
  </svg>
);

const FireIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 23c-3.866 0-7-3.134-7-7 0-2.485 1.336-4.979 3.063-6.932.69-.78 1.468-1.476 2.25-2.068-.023.593.097 1.309.544 1.996.446.687 1.143 1.289 2.143 1.504-.21-1.966.497-3.998 1.819-5.5 1.043 1.934 2.681 3.17 3.181 5.5.5 2.33 0 4.5-1 6s-2 2.5-3 3c1-1 1.5-2 1.5-3.5s-.5-2.5-1.5-3.5c0 2-1 3-2 4s-1.5 2.5-1.5 4c0 1.38.56 2.63 1.465 3.535A4.992 4.992 0 0112 23z"/>
  </svg>
);

const CloseIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const AnalyzeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8" />
    <path d="M21 21l-4.35-4.35" />
  </svg>
);

const SettingsIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

/**
 * Position mapping - only real basketball positions (not roster slots)
 */
const POSITION_MAP = {
  0: 'PG',
  1: 'SG',
  2: 'SF',
  3: 'PF',
  4: 'C'
  // Don't map 5 (G), 6 (F), 7+ (UTIL, BENCH, etc.)
};

/**
 * Convert eligible slots array to position string
 * Only includes real positions (PG, SG, SF, PF, C), not roster slots (G, F, UTIL)
 */
function getPositionString(player) {
  // First check eligible_slots array
  const eligibleSlots = player.eligible_slots || player.eligibleSlots;

  if (eligibleSlots && Array.isArray(eligibleSlots) && eligibleSlots.length > 0) {
    const positions = eligibleSlots
      .filter(slot => slot >= 0 && slot <= 4)
      .map(slot => POSITION_MAP[slot])
      .filter(Boolean);

    if (positions.length > 0) {
      return positions.join('/');
    }
  }

  // Fallback to position field if it's already a string
  if (player.position && typeof player.position === 'string') {
    // If it's a numeric string, try to map it
    const posNum = parseInt(player.position, 10);
    if (!isNaN(posNum) && POSITION_MAP[posNum]) {
      return POSITION_MAP[posNum];
    }
    // If it's already a position abbreviation, return it
    if (['PG', 'SG', 'SF', 'PF', 'C'].includes(player.position.toUpperCase())) {
      return player.position.toUpperCase();
    }
    return player.position;
  }

  return 'UTIL';
}

/**
 * Trade Analyzer Modal Component
 * Supports two-team selection with roster display for each team
 */
function TradeAnalyzerModal({
  isOpen,
  onClose,
  leagueId,
  userTeamId,
  allTeams = [],
  teamRosters = {}, // Map of teamId -> roster array
  leagueAverages = {},
  numTeams = 10,
}) {
  // Team selection state
  const [team1Id, setTeam1Id] = useState(userTeamId || null);
  const [team2Id, setTeam2Id] = useState(null);

  // Players selected from each team
  const [team1PlayersOut, setTeam1PlayersOut] = useState([]);
  const [team2PlayersOut, setTeam2PlayersOut] = useState([]);

  // Analysis state
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Get rosters for selected teams
  const team1Roster = teamRosters[team1Id] || [];
  const team2Roster = teamRosters[team2Id] || [];

  // Reset selections when teams change
  const handleTeam1Change = (newTeamId) => {
    setTeam1Id(newTeamId);
    setTeam1PlayersOut([]);
    setAnalysis(null);
    setError(null);
  };

  const handleTeam2Change = (newTeamId) => {
    setTeam2Id(newTeamId);
    setTeam2PlayersOut([]);
    setAnalysis(null);
    setError(null);
  };

  // Toggle player selection
  const togglePlayer = (playerId, selectedList, setSelectedList, roster) => {
    const player = roster.find(p => p.player_id === playerId);
    if (!player) return;

    if (selectedList.includes(playerId)) {
      setSelectedList(selectedList.filter(id => id !== playerId));
    } else {
      setSelectedList([...selectedList, playerId]);
    }
  };

  // Get team name by ID
  const getTeamName = (teamId) => {
    const team = allTeams.find(t => t.team_id === teamId || t.id === teamId);
    return team?.team_name || team?.name || 'Select Team';
  };

  // Determine which team is "you" for analysis perspective
  const isTeam1User = team1Id === userTeamId;
  const myPlayersOut = isTeam1User ? team1PlayersOut : team2PlayersOut;
  const myPlayersIn = isTeam1User ? team2PlayersOut : team1PlayersOut;
  const myRoster = isTeam1User ? team1Roster : team2Roster;
  const theirRoster = isTeam1User ? team2Roster : team1Roster;

  const handleAnalyze = async () => {
    if (!team1Id || !team2Id) {
      setError('Please select both teams');
      return;
    }
    if (team1PlayersOut.length === 0 || team2PlayersOut.length === 0) {
      setError('Please select at least one player from each team');
      return;
    }

    setLoading(true);
    setError(null);
    setAnalysis(null);

    try {
      // Get full player data for selected players
      const getPlayerData = (playerId, roster) => {
        const player = roster.find(p => p.player_id === playerId);
        return player ? {
          player_id: player.player_id,
          name: player.name || player.player_name,
          nba_team: player.nba_team || player.team,
          per_game_stats: player.per_game_stats || player.stats || {},
          projected_games: player.projected_games || 30,
        } : null;
      };

      // Always send from user's perspective:
      // team1 = players user gives away
      // team2 = players user receives
      const playersOutData = myPlayersOut
        .map(id => getPlayerData(id, myRoster))
        .filter(Boolean);
      const playersInData = myPlayersIn
        .map(id => getPlayerData(id, theirRoster))
        .filter(Boolean);

      const result = await analyzeTrade(leagueId, {
        team1_id: isTeam1User ? team1Id : team2Id,
        team2_id: isTeam1User ? team2Id : team1Id,
        team1_players: myPlayersOut,
        team2_players: myPlayersIn,
        team1_player_data: playersOutData,
        team2_player_data: playersInData,
        league_averages: leagueAverages,
        num_teams: numTeams,
      });

      setAnalysis(result.analysis || result);
    } catch (err) {
      console.error('Trade analysis error:', err);
      setError(err.response?.data?.error || err.message || 'Failed to analyze trade');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setTeam1PlayersOut([]);
    setTeam2PlayersOut([]);
    setAnalysis(null);
    setError(null);
  };

  const handleFullReset = () => {
    setTeam1Id(userTeamId || null);
    setTeam2Id(null);
    handleReset();
  };

  const getRecommendationClass = (recommendation) => {
    switch (recommendation) {
      case 'ACCEPT': return 'recommendation-accept';
      case 'REJECT': return 'recommendation-reject';
      case 'CONSIDER': return 'recommendation-consider';
      case 'COUNTER': return 'recommendation-counter';
      default: return '';
    }
  };

  const getGradeClass = (grade) => {
    if (!grade) return '';
    if (grade.startsWith('A')) return 'grade-a';
    if (grade.startsWith('B')) return 'grade-b';
    if (grade.startsWith('C')) return 'grade-c';
    return 'grade-d';
  };

  if (!isOpen) return null;

  return (
    <div className="trade-modal-overlay" onClick={onClose}>
      <div className="trade-modal" onClick={e => e.stopPropagation()}>
        <div className="trade-modal-header">
          <h2>
            <TradeIcon />
            Trade Analyzer
          </h2>
          <button className="modal-close-btn" onClick={onClose}>
            <CloseIcon />
          </button>
        </div>

        <div className="trade-modal-content">
          {/* Team Selection Dropdowns */}
          <div className="trade-team-selectors">
            <div className="team-selector">
              <label>Team 1 {team1Id === userTeamId && <span className="your-team-badge">(You)</span>}</label>
              <select
                className="team-select"
                value={team1Id || ''}
                onChange={(e) => handleTeam1Change(e.target.value ? parseInt(e.target.value) : null)}
              >
                <option value="">Select team...</option>
                {allTeams.map(team => (
                  <option
                    key={team.team_id || team.id}
                    value={team.team_id || team.id}
                    disabled={team.team_id === team2Id || team.id === team2Id}
                  >
                    {team.team_name || team.name}
                    {(team.team_id === userTeamId || team.id === userTeamId) ? ' (You)' : ''}
                  </option>
                ))}
              </select>
            </div>

            <div className="trade-arrow-small">
              <TradeIcon />
            </div>

            <div className="team-selector">
              <label>Team 2 {team2Id === userTeamId && <span className="your-team-badge">(You)</span>}</label>
              <select
                className="team-select"
                value={team2Id || ''}
                onChange={(e) => handleTeam2Change(e.target.value ? parseInt(e.target.value) : null)}
              >
                <option value="">Select team...</option>
                {allTeams.map(team => (
                  <option
                    key={team.team_id || team.id}
                    value={team.team_id || team.id}
                    disabled={team.team_id === team1Id || team.id === team1Id}
                  >
                    {team.team_name || team.name}
                    {(team.team_id === userTeamId || team.id === userTeamId) ? ' (You)' : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Roster Selection Area */}
          {team1Id && team2Id ? (
            <div className="trade-selection-area">
              {/* Team 1 Roster */}
              <div className="trade-column give">
                <h3>
                  {getTeamName(team1Id)} Gives
                  {team1Id === userTeamId && <span className="column-badge you">You</span>}
                </h3>
                <div className="roster-checklist">
                  {team1Roster.length > 0 ? (
                    team1Roster.map(player => (
                      <label
                        key={player.player_id}
                        className={`roster-player-checkbox ${team1PlayersOut.includes(player.player_id) ? 'selected' : ''}`}
                      >
                        <input
                          type="checkbox"
                          checked={team1PlayersOut.includes(player.player_id)}
                          onChange={() => togglePlayer(
                            player.player_id,
                            team1PlayersOut,
                            setTeam1PlayersOut,
                            team1Roster
                          )}
                        />
                        <span className="player-info">
                          <strong>{player.name || player.player_name}</strong>
                          <span className="player-meta">
                            {getPositionString(player)} • {player.nba_team || player.team}
                          </span>
                        </span>
                        {player.z_score_value !== undefined && (
                          <span className={`player-value ${player.z_score_value >= 0 ? 'positive' : 'negative'}`}>
                            {player.z_score_value >= 0 ? '+' : ''}{player.z_score_value?.toFixed(1)}
                          </span>
                        )}
                      </label>
                    ))
                  ) : (
                    <div className="no-roster-hint">No roster data available</div>
                  )}
                </div>
                {team1PlayersOut.length > 0 && (
                  <div className="selected-count">
                    {team1PlayersOut.length} player{team1PlayersOut.length !== 1 ? 's' : ''} selected
                  </div>
                )}
              </div>

              {/* Trade Arrow */}
              <div className="trade-arrow">
                <TradeIcon />
              </div>

              {/* Team 2 Roster */}
              <div className="trade-column receive">
                <h3>
                  {getTeamName(team2Id)} Gives
                  {team2Id === userTeamId && <span className="column-badge you">You</span>}
                </h3>
                <div className="roster-checklist">
                  {team2Roster.length > 0 ? (
                    team2Roster.map(player => (
                      <label
                        key={player.player_id}
                        className={`roster-player-checkbox ${team2PlayersOut.includes(player.player_id) ? 'selected' : ''}`}
                      >
                        <input
                          type="checkbox"
                          checked={team2PlayersOut.includes(player.player_id)}
                          onChange={() => togglePlayer(
                            player.player_id,
                            team2PlayersOut,
                            setTeam2PlayersOut,
                            team2Roster
                          )}
                        />
                        <span className="player-info">
                          <strong>{player.name || player.player_name}</strong>
                          <span className="player-meta">
                            {getPositionString(player)} • {player.nba_team || player.team}
                          </span>
                        </span>
                        {player.z_score_value !== undefined && (
                          <span className={`player-value ${player.z_score_value >= 0 ? 'positive' : 'negative'}`}>
                            {player.z_score_value >= 0 ? '+' : ''}{player.z_score_value?.toFixed(1)}
                          </span>
                        )}
                      </label>
                    ))
                  ) : (
                    <div className="no-roster-hint">No roster data available</div>
                  )}
                </div>
                {team2PlayersOut.length > 0 && (
                  <div className="selected-count">
                    {team2PlayersOut.length} player{team2PlayersOut.length !== 1 ? 's' : ''} selected
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="trade-selection-placeholder">
              <p>Select both teams above to view their rosters and build a trade</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="trade-actions">
            <button
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={loading || !team1Id || !team2Id || team1PlayersOut.length === 0 || team2PlayersOut.length === 0}
            >
              {loading ? (
                <span className="loading-spinner">Analyzing...</span>
              ) : (
                <>
                  <AnalyzeIcon />
                  Analyze Trade
                </>
              )}
            </button>
            <button className="reset-btn" onClick={handleReset} disabled={loading}>
              Clear Players
            </button>
            <button className="reset-btn secondary" onClick={handleFullReset} disabled={loading}>
              Reset All
            </button>
          </div>

          {/* User Perspective Indicator */}
          {team1Id && team2Id && userTeamId && (team1Id === userTeamId || team2Id === userTeamId) && (
            <div className="perspective-indicator">
              Analysis shown from <strong>{getTeamName(userTeamId)}'s</strong> perspective
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="trade-error">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {/* Analysis Results */}
          {analysis && (
            <div className="trade-analysis-results">
              <div className="analysis-header">
                <div className={`recommendation-badge ${getRecommendationClass(analysis.recommendation)}`}>
                  {analysis.recommendation}
                </div>
                <div className={`grade-badge ${getGradeClass(analysis.trade_grade)}`}>
                  {analysis.trade_grade}
                </div>
              </div>

              <div className="analysis-summary">
                <div className="summary-item z-score">
                  <span className="summary-label">Z-Score Change</span>
                  <span className={`summary-value ${analysis.net_z_score_change >= 0 ? 'positive' : 'negative'}`}>
                    {analysis.net_z_score_change >= 0 ? '+' : ''}{analysis.net_z_score_change?.toFixed(2)}/game
                  </span>
                </div>
                <div className="summary-item fairness">
                  <span className="summary-label">Fairness</span>
                  <span className={`summary-value ${analysis.fairness_score >= 0 ? 'positive' : 'negative'}`}>
                    {analysis.fairness_score >= 0 ? '+' : ''}{analysis.fairness_score?.toFixed(1)}
                  </span>
                </div>
              </div>

              <div className="analysis-reason">
                <span className="reason-icon">💡</span>
                <p>{analysis.reason}</p>
              </div>

              {/* Category Impact Table */}
              {analysis.category_changes && Object.keys(analysis.category_changes).length > 0 && (
                <div className="category-impact-table">
                  <h4>Category Impact (per game)</h4>
                  <table className="impact-table">
                    <thead>
                      <tr>
                        <th>Category</th>
                        <th>Change</th>
                        <th>Impact</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(analysis.category_changes)
                        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])) // Sort by magnitude
                        .map(([cat, change]) => {
                          const isPositive = change > 0.05;
                          const isNegative = change < -0.05;
                          const colorClass = isPositive ? 'positive' : isNegative ? 'negative' : 'neutral';
                          const arrow = isPositive ? '↑' : isNegative ? '↓' : '→';

                          return (
                            <tr key={cat} className={`impact-row ${colorClass}`}>
                              <td className="cat-name">{cat.toUpperCase()}</td>
                              <td className="cat-change">
                                {change > 0 ? '+' : ''}{change.toFixed(2)}
                              </td>
                              <td className="cat-arrow">{arrow}</td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Impact score badge with color coding
 */
function ImpactBadge({ score }) {
  const getScoreClass = (score) => {
    if (score >= 80) return 'impact-excellent';
    if (score >= 60) return 'impact-good';
    if (score >= 40) return 'impact-average';
    return 'impact-low';
  };

  return (
    <div className={`impact-badge ${getScoreClass(score)}`}>
      <span className="impact-value">{score}</span>
      <span className="impact-label">Impact</span>
    </div>
  );
}

/**
 * Waiver Wire Targets Section with Click-to-Analyze
 */
function WaiverTargets({
  targets = [],
  maxItems = 3,
  leagueId,
  currentRoster = [],
  leagueAverages = {},
}) {
  const [expanded, setExpanded] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const displayTargets = expanded ? targets : targets.slice(0, maxItems);
  const hasMore = targets.length > maxItems;

  /**
   * Handle clicking on a waiver target to analyze
   */
  const handleAnalyzePlayer = async (player) => {
    // If already selected, toggle off
    if (selectedPlayer?.player_id === player.player_id || selectedPlayer?.id === player.id) {
      setSelectedPlayer(null);
      setAnalysis(null);
      return;
    }

    // Check if we have the data needed for analysis
    if (!leagueId || !currentRoster || currentRoster.length === 0) {
      setSelectedPlayer(player);
      setAnalysis(null);
      setError('Roster data not available for analysis');
      return;
    }

    try {
      setLoading(true);
      setError('');
      setSelectedPlayer(player);

      // Build player data for analysis
      const playerToAdd = {
        player_id: player.player_id || player.id,
        name: player.name || player.player_name,
        nba_team: player.nba_team || player.team,
        position: player.position,
        per_game_stats: player.per_game_stats || player.stats || {},
        projected_games: player.projected_games || 30,
        eligible_slots: player.eligible_slots || player.eligibleSlots || [],
      };

      const response = await analyzeWaiver(leagueId, {
        player_to_add: playerToAdd,
        current_roster: currentRoster,
        league_averages: leagueAverages,
      });

      setAnalysis(response.analysis || response);
    } catch (err) {
      console.error('Failed to analyze waiver:', err);
      setError(err.response?.data?.error || 'Failed to analyze waiver move');
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Get color class for recommendation
   */
  const getRecommendationClass = (recommendation) => {
    switch (recommendation) {
      case 'ADD': return 'recommendation-accept';
      case 'PASS': return 'recommendation-reject';
      case 'CONSIDER': return 'recommendation-consider';
      default: return '';
    }
  };

  /**
   * Get color class for grade
   */
  const getGradeClass = (grade) => {
    if (!grade) return '';
    if (grade.startsWith('A')) return 'grade-a';
    if (grade.startsWith('B')) return 'grade-b';
    if (grade.startsWith('C')) return 'grade-c';
    return 'grade-d';
  };

  if (!targets || targets.length === 0) {
    return (
      <div className="insight-section">
        <div className="insight-header">
          <WaiverIcon />
          <h3>Top Waiver Targets</h3>
        </div>
        <div className="insight-empty">
          <p>No waiver recommendations available</p>
          <span className="empty-hint">Check back after more games are played</span>
        </div>
      </div>
    );
  }

  return (
    <div className="insight-section">
      <div className="insight-header">
        <WaiverIcon />
        <h3>Top Waiver Targets</h3>
        {hasMore && (
          <div className="header-actions">
            <button
              className="expand-btn"
              onClick={() => setExpanded(!expanded)}
              title={expanded ? 'Show less' : `Show all ${targets.length} targets`}
            >
              {expanded ? 'Show Less' : `+${targets.length - maxItems} more`}
            </button>
          </div>
        )}
      </div>

      <ul className={`waiver-list ${expanded ? 'expanded' : ''}`}>
        {displayTargets.map((player, idx) => {
          const playerId = player.player_id || player.id;
          const isSelected = selectedPlayer && (selectedPlayer.player_id === playerId || selectedPlayer.id === playerId);

          return (
            <li
              key={playerId || idx}
              className={`waiver-item ${isSelected ? 'selected' : ''} ${leagueId ? 'clickable' : ''}`}
              onClick={() => leagueId && handleAnalyzePlayer(player)}
              title={leagueId ? 'Click to analyze add/drop' : ''}
            >
              <div className="waiver-rank">{idx + 1}</div>
              <div className="waiver-info">
                <div className="waiver-player">
                  <span className="player-name">{player.name}</span>
                  <span className="player-meta">
                    {getPositionString(player)} • {player.nba_team || player.team}
                  </span>
                </div>
                <div className="waiver-reason">
                  {player.trending === 'up' && <TrendUpIcon />}
                  {player.trending === 'down' && <TrendDownIcon />}
                  {player.hot && <FireIcon />}
                  <span>{player.reason || 'Strong pickup candidate'}</span>
                </div>
              </div>
              <ImpactBadge score={player.impact_score || player.impact || 0} />
            </li>
          );
        })}
      </ul>

      {/* Loading Indicator */}
      {loading && (
        <div className="waiver-analysis-loading">
          <span className="loading-spinner-small"></span>
          Analyzing...
        </div>
      )}

      {/* Error Message */}
      {error && !loading && selectedPlayer && (
        <div className="waiver-analysis-error">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}

      {/* Analysis Results */}
      {analysis && selectedPlayer && !loading && (
        <div className="waiver-analysis-results">
          <div className="analysis-header">
            <div className="analysis-title">
              <span className="analysis-action">Add</span>
              <strong>{selectedPlayer.name}</strong>
            </div>
            <div className="analysis-badges">
              <span className={`recommendation-badge ${getRecommendationClass(analysis.recommendation)}`}>
                {analysis.recommendation}
              </span>
              <span className={`grade-badge ${getGradeClass(analysis.grade)}`}>
                {analysis.grade}
              </span>
              <button
                className="close-analysis-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedPlayer(null);
                  setAnalysis(null);
                }}
                title="Close analysis"
              >
                <CloseIcon />
              </button>
            </div>
          </div>

          <div className="analysis-details">
            {/* Drop Recommendation */}
            {analysis.drop_player_name && analysis.drop_player_name !== 'No droppable player' && (
              <div className="analysis-row drop-row">
                <span className="row-label">Drop:</span>
                <span className="row-value">
                  <strong>{analysis.drop_player_name}</strong>
                  <span className={`z-score ${analysis.drop_player_z_score >= 0 ? 'positive' : 'negative'}`}>
                    ({analysis.drop_player_z_score >= 0 ? '+' : ''}{analysis.drop_player_z_score?.toFixed(2)})
                  </span>
                </span>
              </div>
            )}

            {/* Net Z-Score Change */}
            <div className="analysis-row z-change-row">
              <span className="row-label">Net Z-Score:</span>
              <span className={`row-value z-change ${analysis.net_z_score_change >= 0 ? 'positive' : 'negative'}`}>
                {analysis.net_z_score_change >= 0 ? '+' : ''}{analysis.net_z_score_change?.toFixed(2)}/game
              </span>
            </div>

            {/* Category Changes */}
            {(analysis.improves_categories?.length > 0 || analysis.hurts_categories?.length > 0) && (
              <div className="analysis-categories">
                {analysis.improves_categories?.length > 0 && (
                  <div className="category-row improves">
                    <span className="category-label">↑</span>
                    <div className="category-chips">
                      {analysis.improves_categories.map((cat, i) => (
                        <span key={i} className="category-chip improves">{cat}</span>
                      ))}
                    </div>
                  </div>
                )}
                {analysis.hurts_categories?.length > 0 && (
                  <div className="category-row hurts">
                    <span className="category-label">↓</span>
                    <div className="category-chips">
                      {analysis.hurts_categories.map((cat, i) => (
                        <span key={i} className="category-chip hurts">{cat}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Reason */}
            {analysis.reason && (
              <div className="analysis-reason">
                <span className="reason-icon">💡</span>
                <span className="reason-text">{analysis.reason}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Trade Opportunities Section
 */
function TradeOpportunities({
  opportunities = [],
  maxItems = 2,
  leagueId,
  userTeamId,
  allTeams = [],
  teamRosters = {},
  leagueAverages = {},
  myCurrentRank = 5,
  myCurrentRotoPoints = 50,
  numTeams = 10,
  showAnalyzer = true,
  onSettingsChange = null,
}) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [tradeMode, setTradeMode] = useState('normal');
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState('');

  // Load trade settings on mount
  useEffect(() => {
    if (leagueId) {
      setSettingsLoading(true);
      fetchTradeSettings(leagueId)
        .then(data => {
          setTradeMode(data.trade_suggestion_mode || 'normal');
          setSettingsError('');
        })
        .catch(err => {
          console.error('Error loading trade settings:', err);
          setSettingsError('Failed to load settings');
        })
        .finally(() => setSettingsLoading(false));
    }
  }, [leagueId]);

  // Handle mode change and save
  const handleModeChange = async (newMode) => {
    setTradeMode(newMode);
    setSettingsSaving(true);
    setSettingsError('');

    try {
      await updateTradeSettings(leagueId, { trade_suggestion_mode: newMode });
      // Close settings panel and trigger refresh
      setShowSettings(false);
      if (onSettingsChange) {
        onSettingsChange({ trade_suggestion_mode: newMode });
      }
    } catch (err) {
      console.error('Error saving trade settings:', err);
      setSettingsError('Failed to save');
    } finally {
      setSettingsSaving(false);
    }
  };

  // Show limited or all suggestions based on expanded state
  const displayOpportunities = expanded ? opportunities : opportunities.slice(0, maxItems);
  const hasMore = opportunities.length > maxItems;

  const modeOptions = [
    { value: 'conservative', label: 'Conservative', hint: '-0.5 to +0.5' },
    { value: 'normal', label: 'Normal', hint: '-0.25 to +1.0' },
    { value: 'aggressive', label: 'Aggressive', hint: '0.0 to +1.5' },
  ];

  return (
    <div className="insight-section">
      <div className="insight-header">
        <TradeIcon />
        <h3>Trade Opportunities</h3>
        <div className="header-actions">
          {leagueId && (
            <button
              className={`settings-btn ${showSettings ? 'active' : ''}`}
              onClick={() => setShowSettings(!showSettings)}
              title="Trade suggestion settings"
            >
              <SettingsIcon />
            </button>
          )}
          {hasMore && (
            <button
              className="expand-btn"
              onClick={() => setExpanded(!expanded)}
              title={expanded ? 'Show less' : `Show all ${opportunities.length} suggestions`}
            >
              {expanded ? 'Show Less' : `+${opportunities.length - maxItems} more`}
            </button>
          )}
          {showAnalyzer && (
            <button
              className="analyze-trade-btn"
              onClick={() => setIsModalOpen(true)}
              title="Analyze a custom trade"
            >
              <AnalyzeIcon />
              Analyze Trade
            </button>
          )}
        </div>
      </div>

      {/* Inline Settings Panel */}
      {showSettings && (
        <div className="trade-settings-inline">
          <div className="settings-label">
            <span>Suggestion Mode</span>
            {settingsLoading && <span className="settings-loading">Loading...</span>}
            {settingsSaving && <span className="settings-saving">Saving...</span>}
            {settingsError && <span className="settings-error">{settingsError}</span>}
          </div>
          <div className="mode-options">
            {modeOptions.map(option => (
              <label
                key={option.value}
                className={`mode-option ${tradeMode === option.value ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="trade_mode"
                  value={option.value}
                  checked={tradeMode === option.value}
                  onChange={() => handleModeChange(option.value)}
                  disabled={settingsLoading || settingsSaving}
                />
                <span className="mode-label">{option.label}</span>
                <span className="mode-hint">{option.hint}</span>
              </label>
            ))}
          </div>
          <p className="settings-hint">
            Changes apply on next refresh. Conservative = fair trades only, Aggressive = trades that benefit you.
          </p>
        </div>
      )}

      {(!opportunities || opportunities.length === 0) ? (
        <div className="insight-empty">
          <p>No trade opportunities identified</p>
          <span className="empty-hint">Use the analyzer to evaluate custom trades</span>
        </div>
      ) : (
        <>
          <ul className={`trade-list ${expanded ? 'expanded' : ''}`}>
            {displayOpportunities.map((trade, idx) => (
              <li key={idx} className="trade-item">
                <div className="trade-content">
                  <div className="trade-partner">
                    <span className="partner-label">Target from</span>
                    <span className="partner-name">{trade.target_team || trade.partner_team}</span>
                  </div>
                  <div className="trade-players">
                    {trade.target_player && (
                      <span className="trade-target">
                        Get: <strong>{trade.target_player}</strong>
                      </span>
                    )}
                    {trade.give_player && (
                      <span className="trade-give">
                        Give: <strong>{trade.give_player}</strong>
                      </span>
                    )}
                  </div>
                  <div className="trade-benefit">
                    <span className="benefit-icon">💡</span>
                    <span className="benefit-text">{trade.reason || trade.benefit}</span>
                  </div>
                  {(trade.improves_categories?.length > 0 || trade.hurts_categories?.length > 0) && (
                    <div className="trade-categories-container">
                      {trade.improves_categories?.length > 0 && (
                        <div className="trade-categories improves-row">
                          <span className="categories-label">↑</span>
                          {trade.improves_categories.map((cat, i) => (
                            <span key={i} className="category-chip improves">{cat}</span>
                          ))}
                        </div>
                      )}
                      {trade.hurts_categories?.length > 0 && (
                        <div className="trade-categories hurts-row">
                          <span className="categories-label">↓</span>
                          {trade.hurts_categories.map((cat, i) => (
                            <span key={i} className="category-chip hurts">{cat}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                {trade.value_gain !== undefined && (
                  <div className={`trade-value ${trade.value_gain >= 0 ? 'positive' : 'negative'}`}>
                    <span className="value-number">
                      {trade.value_gain >= 0 ? '+' : ''}{typeof trade.value_gain === 'number' ? trade.value_gain.toFixed(1) : trade.value_gain}
                    </span>
                    <span className="value-label">Z-Score</span>
                  </div>
                )}
              </li>
            ))}
          </ul>
          {!expanded && hasMore && (
            <div className="show-more-hint">
              <button
                className="show-more-btn"
                onClick={() => setExpanded(true)}
              >
                Show {opportunities.length - maxItems} more suggestion{opportunities.length - maxItems > 1 ? 's' : ''}
              </button>
            </div>
          )}
        </>
      )}

      {/* Trade Analyzer Modal */}
      <TradeAnalyzerModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        leagueId={leagueId}
        userTeamId={userTeamId}
        allTeams={allTeams}
        teamRosters={teamRosters}
        leagueAverages={leagueAverages}
        myCurrentRank={myCurrentRank}
        myCurrentRotoPoints={myCurrentRotoPoints}
        numTeams={numTeams}
      />
    </div>
  );
}

/**
 * Category Analysis Section
 */
function CategoryAnalysis({ analysis = {} }) {
  const { strengths = [], weaknesses = [], neutral = [] } = analysis;

  if (strengths.length === 0 && weaknesses.length === 0) {
    return (
      <div className="insight-section">
        <div className="insight-header">
          <CategoryIcon />
          <h3>Category Analysis</h3>
        </div>
        <div className="insight-empty">
          <p>Category analysis not available</p>
          <span className="empty-hint">Add a league to see your strengths and weaknesses</span>
        </div>
      </div>
    );
  }

  return (
    <div className="insight-section">
      <div className="insight-header">
        <CategoryIcon />
        <h3>Category Analysis</h3>
      </div>
      <div className="category-grid">
        {/* Strengths */}
        {strengths.length > 0 && (
          <div className="category-group strengths">
            <span className="category-group-label">
              <span className="indicator strength">✓</span>
              Strengths
            </span>
            <div className="category-tags">
              {strengths.map((cat, idx) => (
                <span key={idx} className="category-tag strength">
                  {cat}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Weaknesses */}
        {weaknesses.length > 0 && (
          <div className="category-group weaknesses">
            <span className="category-group-label">
              <span className="indicator weakness">!</span>
              Needs Work
            </span>
            <div className="category-tags">
              {weaknesses.map((cat, idx) => (
                <span key={idx} className="category-tag weakness">
                  {cat}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Neutral/Average */}
        {neutral.length > 0 && (
          <div className="category-group neutral">
            <span className="category-group-label">
              <span className="indicator neutral">—</span>
              Average
            </span>
            <div className="category-tags">
              {neutral.map((cat, idx) => (
                <span key={idx} className="category-tag neutral">
                  {cat}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Format rank with ordinal suffix
 */
const formatOrdinal = (rank) => {
  if (typeof rank !== 'number') return rank;
  const suffixes = ['th', 'st', 'nd', 'rd'];
  const v = rank % 100;
  return rank + (suffixes[(v - 20) % 10] || suffixes[v] || suffixes[0]);
};

/**
 * Category Movements Section - Roto-specific
 * Shows category rank changes (e.g., "You're projected to drop from 2nd to 4th in Assists")
 */
function CategoryMovements({ movements = [], maxItems = 4 }) {
  if (!movements || movements.length === 0) {
    return (
      <div className="insight-section category-movements">
        <div className="insight-header">
          <RotoIcon />
          <h3>Category Projections</h3>
        </div>
        <div className="insight-empty">
          <p>No significant category changes projected</p>
          <span className="empty-hint">Your category ranks are stable</span>
        </div>
      </div>
    );
  }

  // Sort by absolute movement (most significant first)
  const sortedMovements = [...movements]
    .filter(m => m.movement !== 0)
    .sort((a, b) => Math.abs(b.movement) - Math.abs(a.movement))
    .slice(0, maxItems);

  const improving = sortedMovements.filter(m => m.movement > 0);
  const declining = sortedMovements.filter(m => m.movement < 0);

  return (
    <div className="insight-section category-movements">
      <div className="insight-header">
        <RotoIcon />
        <h3>Category Projections</h3>
      </div>
      <div className="movements-content">
        {/* Improving Categories */}
        {improving.length > 0 && (
          <div className="movement-group improving">
            <span className="movement-group-label">
              <ArrowUpIcon />
              Improving
            </span>
            <ul className="movement-list">
              {improving.map((cat, idx) => (
                <li key={idx} className="movement-item improving">
                  <span className="movement-category">{cat.category}</span>
                  <span className="movement-detail">
                    {formatOrdinal(cat.currentRank)} → {formatOrdinal(cat.projectedRank)}
                  </span>
                  <span className="movement-badge improving">+{cat.movement}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Declining Categories */}
        {declining.length > 0 && (
          <div className="movement-group declining">
            <span className="movement-group-label">
              <ArrowDownIcon />
              Declining
            </span>
            <ul className="movement-list">
              {declining.map((cat, idx) => (
                <li key={idx} className="movement-item declining">
                  <span className="movement-category">{cat.category}</span>
                  <span className="movement-detail">
                    {formatOrdinal(cat.currentRank)} → {formatOrdinal(cat.projectedRank)}
                  </span>
                  <span className="movement-badge declining">{cat.movement}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {improving.length === 0 && declining.length === 0 && (
          <div className="insight-empty">
            <p>All categories projected to stay the same</p>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Main QuickInsights component
 */
function QuickInsights({
  waiverTargets = [],
  tradeOpportunities = [],
  categoryAnalysis = {},
  categoryMovements = [],
  title = 'Quick Insights',
  showWaivers = true,
  showTrades = true,
  showCategories = true,
  showMovements = false,
  isRoto = false,
  compact = false,
  // Trade analyzer props
  leagueId,
  userTeamId,
  allTeams = [],
  teamRosters = {},
  leagueAverages = {},
  myCurrentRank = 5,
  myCurrentRotoPoints = 50,
  numTeams = 10,
  showTradeAnalyzer = true,
  onSettingsChange = null,
}) {
  // Get user's current roster for waiver analysis
  const currentRoster = userTeamId ? (teamRosters[userTeamId] || []) : [];

  return (
    <div className={`quick-insights ${compact ? 'compact' : ''} ${isRoto ? 'roto-mode' : ''}`}>
      {title && <h2 className="insights-title">{title}</h2>}

      <div className="insights-content">
        {/* Show category movements first for Roto leagues */}
        {(showMovements || isRoto) && categoryMovements.length > 0 && (
          <CategoryMovements movements={categoryMovements} maxItems={compact ? 3 : 4} />
        )}

        {showWaivers && (
          <WaiverTargets
            targets={waiverTargets}
            maxItems={compact ? 2 : 3}
            leagueId={leagueId}
            currentRoster={currentRoster}
            leagueAverages={leagueAverages}
          />
        )}

        {showTrades && (
          <TradeOpportunities
            opportunities={tradeOpportunities}
            maxItems={compact ? 1 : 2}
            leagueId={leagueId}
            userTeamId={userTeamId}
            allTeams={allTeams}
            teamRosters={teamRosters}
            leagueAverages={leagueAverages}
            myCurrentRank={myCurrentRank}
            myCurrentRotoPoints={myCurrentRotoPoints}
            numTeams={numTeams}
            showAnalyzer={showTradeAnalyzer}
            onSettingsChange={onSettingsChange}
          />
        )}

        {showCategories && (
          <CategoryAnalysis analysis={categoryAnalysis} />
        )}
      </div>
    </div>
  );
}

QuickInsights.propTypes = {
  waiverTargets: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      name: PropTypes.string.isRequired,
      position: PropTypes.string,
      nba_team: PropTypes.string,
      team: PropTypes.string,
      impact_score: PropTypes.number,
      impact: PropTypes.number,
      reason: PropTypes.string,
      trending: PropTypes.oneOf(['up', 'down', null]),
      hot: PropTypes.bool,
    })
  ),
  tradeOpportunities: PropTypes.arrayOf(
    PropTypes.shape({
      target_team: PropTypes.string,
      partner_team: PropTypes.string,
      target_player: PropTypes.string,
      give_player: PropTypes.string,
      reason: PropTypes.string,
      benefit: PropTypes.string,
      value_gain: PropTypes.number,
    })
  ),
  categoryAnalysis: PropTypes.shape({
    strengths: PropTypes.arrayOf(PropTypes.string),
    weaknesses: PropTypes.arrayOf(PropTypes.string),
    neutral: PropTypes.arrayOf(PropTypes.string),
  }),
  categoryMovements: PropTypes.arrayOf(
    PropTypes.shape({
      category: PropTypes.string.isRequired,
      currentRank: PropTypes.number.isRequired,
      projectedRank: PropTypes.number.isRequired,
      movement: PropTypes.number.isRequired,
    })
  ),
  title: PropTypes.string,
  showWaivers: PropTypes.bool,
  showTrades: PropTypes.bool,
  showCategories: PropTypes.bool,
  showMovements: PropTypes.bool,
  isRoto: PropTypes.bool,
  compact: PropTypes.bool,
  // Trade analyzer props
  leagueId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  userTeamId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  allTeams: PropTypes.array,
  teamRosters: PropTypes.object,
  leagueAverages: PropTypes.object,
  myCurrentRank: PropTypes.number,
  myCurrentRotoPoints: PropTypes.number,
  numTeams: PropTypes.number,
  showTradeAnalyzer: PropTypes.bool,
  onSettingsChange: PropTypes.func,
};

// Export sub-components for direct use
QuickInsights.WaiverTargets = WaiverTargets;
QuickInsights.TradeOpportunities = TradeOpportunities;
QuickInsights.CategoryAnalysis = CategoryAnalysis;
QuickInsights.CategoryMovements = CategoryMovements;

export default QuickInsights;

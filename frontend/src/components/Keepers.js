import React, { useState, useEffect, useCallback } from 'react';
import { fetchKeepers } from '../services/api';

const Keepers = ({ leagueId }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [keeperData, setKeeperData] = useState(null);
  const [sortBy, setSortBy] = useState('rank');
  const [sortDesc, setSortDesc] = useState(false);
  const [selectedTeamId, setSelectedTeamId] = useState(null);

  const loadKeepers = useCallback(async (teamId = null) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchKeepers(leagueId, teamId);
      setKeeperData(data);
      // Set selected team ID if not already set
      if (!selectedTeamId && data.team_id) {
        setSelectedTeamId(data.team_id);
      }
    } catch (err) {
      console.error('Error loading keepers:', err);
      setError(err.response?.data?.error || 'Failed to load keeper data');
    } finally {
      setLoading(false);
    }
  }, [leagueId, selectedTeamId]);

  useEffect(() => {
    loadKeepers();
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const handleTeamChange = (e) => {
    const newTeamId = parseInt(e.target.value, 10);
    setSelectedTeamId(newTeamId);
    loadKeepers(newTeamId);
  };

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortDesc(!sortDesc);
    } else {
      setSortBy(column);
      // Default to descending for z-scores and games played (higher is better)
      setSortDesc(column === 'total_z_score' || column === 'games_played' || column.startsWith('cat_'));
    }
  };

  const getSortedPlayers = () => {
    if (!keeperData?.players) return [];

    return [...keeperData.players].sort((a, b) => {
      let aVal, bVal;

      // Check if sorting by a category z-score
      if (sortBy.startsWith('cat_')) {
        const cat = sortBy.replace('cat_', '');
        aVal = a.category_z_scores?.[cat] ?? a.category_z_scores?.[cat.toLowerCase()];
        bVal = b.category_z_scores?.[cat] ?? b.category_z_scores?.[cat.toLowerCase()];
      } else {
        aVal = a[sortBy];
        bVal = b[sortBy];
      }

      // Handle null/undefined
      if (aVal == null) aVal = sortDesc ? -Infinity : Infinity;
      if (bVal == null) bVal = sortDesc ? -Infinity : Infinity;

      // Numeric comparison
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDesc ? bVal - aVal : aVal - bVal;
      }

      // String comparison
      aVal = String(aVal).toLowerCase();
      bVal = String(bVal).toLowerCase();
      return sortDesc ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
    });
  };

  const formatZScore = (z) => {
    if (z == null) return '-';
    return z >= 0 ? `+${z.toFixed(2)}` : z.toFixed(2);
  };

  const getZScoreClass = (z) => {
    if (z == null) return '';
    if (z >= 2) return 'z-excellent';
    if (z >= 1) return 'z-good';
    if (z >= 0) return 'z-average';
    if (z >= -1) return 'z-below';
    return 'z-poor';
  };

  const getInjuryBadge = (status) => {
    if (!status || status === 'ACTIVE') return null;
    const badgeClass = status === 'OUT' ? 'injury-badge out' :
                       status === 'DAY_TO_DAY' ? 'injury-badge dtd' :
                       'injury-badge';
    const label = status === 'DAY_TO_DAY' ? 'DTD' : status;
    return <span className={badgeClass}>{label}</span>;
  };

  if (loading) {
    return (
      <div className="keepers-container">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading keeper data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="keepers-container">
        <div className="error-message">{error}</div>
      </div>
    );
  }

  if (!keeperData?.is_keeper_league) {
    return (
      <div className="keepers-container">
        <div className="keepers-not-available">
          <h3>Not a Keeper League</h3>
          <p>This league is not configured as a keeper league in ESPN.</p>
        </div>
      </div>
    );
  }

  const sortedPlayers = getSortedPlayers();
  const categories = keeperData.categories || [];
  const teams = keeperData.teams || [];

  return (
    <div className="keepers-container">
      <div className="keepers-header">
        <h2>Keeper Rankings</h2>
        <div className="keeper-info">
          <span className="keeper-count">
            Keepers Allowed: <strong>{keeperData.keeper_count}</strong>
          </span>
        </div>
      </div>

      <div className="keepers-description">
        <p>
          Players ranked by z-score value. Consider age and games played for keeper value.
          Younger players with high z-scores offer better long-term value.
        </p>
      </div>

      <div className="team-selector-row">
        <select
          id="team-select"
          value={selectedTeamId || ''}
          onChange={handleTeamChange}
          className="team-dropdown"
        >
          {teams.map(team => (
            <option key={team.team_id} value={team.team_id}>
              {team.team_name}{team.is_user_team ? ' (You)' : ''}
            </option>
          ))}
        </select>
      </div>

      <div className="keepers-table-container">
        <table className="keepers-table">
          <thead>
            <tr>
              <th
                className={`sortable ${sortBy === 'rank' ? 'sorted' : ''}`}
                onClick={() => handleSort('rank')}
              >
                Rank {sortBy === 'rank' && (sortDesc ? '↓' : '↑')}
              </th>
              <th
                className={`sortable ${sortBy === 'name' ? 'sorted' : ''}`}
                onClick={() => handleSort('name')}
              >
                Player {sortBy === 'name' && (sortDesc ? '↓' : '↑')}
              </th>
              <th>Pos</th>
              <th>Team</th>
              <th
                className={`sortable ${sortBy === 'age' ? 'sorted' : ''}`}
                onClick={() => handleSort('age')}
              >
                Age {sortBy === 'age' && (sortDesc ? '↓' : '↑')}
              </th>
              <th
                className={`sortable ${sortBy === 'games_played' ? 'sorted' : ''}`}
                onClick={() => handleSort('games_played')}
              >
                GP {sortBy === 'games_played' && (sortDesc ? '↓' : '↑')}
              </th>
              <th
                className={`sortable ${sortBy === 'total_z_score' ? 'sorted' : ''}`}
                onClick={() => handleSort('total_z_score')}
              >
                Z-Score {sortBy === 'total_z_score' && (sortDesc ? '↓' : '↑')}
              </th>
              {categories.map(cat => (
                <th
                  key={cat}
                  className={`category-header sortable ${sortBy === `cat_${cat}` ? 'sorted' : ''}`}
                  onClick={() => handleSort(`cat_${cat}`)}
                >
                  {cat} {sortBy === `cat_${cat}` && (sortDesc ? '↓' : '↑')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedPlayers.map((player, index) => (
              <tr
                key={player.player_id || index}
                className={index < keeperData.keeper_count ? 'keeper-candidate' : ''}
              >
                <td className="rank-cell">{player.rank}</td>
                <td className="player-name-cell">
                  {player.name}
                  {getInjuryBadge(player.injury_status)}
                </td>
                <td>{player.eligible_positions?.join(', ') || player.position}</td>
                <td>{player.nba_team}</td>
                <td className={player.age && player.age <= 25 ? 'young-player' : player.age && player.age >= 32 ? 'old-player' : ''}>
                  {player.age || '-'}
                </td>
                <td>{player.games_played}</td>
                <td className={`z-score-cell ${getZScoreClass(player.total_z_score)}`}>
                  {formatZScore(player.total_z_score)}
                </td>
                {categories.map(cat => {
                  // Try both the original category name and lowercase version
                  const catZ = player.category_z_scores?.[cat] ?? player.category_z_scores?.[cat.toLowerCase()];
                  return (
                    <td key={cat} className={`z-score-cell ${getZScoreClass(catZ)}`}>
                      {formatZScore(catZ)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="keepers-legend">
        <h4>Keeper Value Guide</h4>
        <div className="legend-items">
          <span className="legend-item">
            <span className="legend-color keeper-candidate-color"></span>
            Recommended Keepers (Top {keeperData.keeper_count})
          </span>
        </div>
        <div className="z-score-legend">
          <span className="legend-item"><span className="z-excellent">+2.0+</span> Elite</span>
          <span className="legend-item"><span className="z-good">+1.0</span> Good</span>
          <span className="legend-item"><span className="z-average">0.0</span> Average</span>
          <span className="legend-item"><span className="z-below">-1.0</span> Below Avg</span>
          <span className="legend-item"><span className="z-poor">-2.0-</span> Poor</span>
        </div>
      </div>
    </div>
  );
};

export default Keepers;

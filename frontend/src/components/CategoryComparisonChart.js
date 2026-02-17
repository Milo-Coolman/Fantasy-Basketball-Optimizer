import React, { useState, useMemo } from 'react';
import PropTypes from 'prop-types';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
  ReferenceLine,
} from 'recharts';
import { useTheme } from '../context/ThemeContext';


/**
 * Theme-aware chart colors
 */
const getChartColors = (isDark) => ({
  grid: isDark ? '#334155' : '#e2e8f0',
  axis: isDark ? '#94a3b8' : '#64748b',
  tooltip: {
    bg: isDark ? '#1e293b' : '#ffffff',
    border: isDark ? '#334155' : '#e2e8f0',
    text: isDark ? '#f8fafc' : '#0f172a',
    subtext: isDark ? '#94a3b8' : '#64748b',
  },
  current: '#3b82f6',      // Blue for current
  projected: '#10b981',    // Green for projected
  improving: '#10b981',    // Green
  declining: '#ef4444',    // Red
  neutral: '#8b5cf6',      // Purple
  top3: '#10b981',         // Green - winning
  middle: '#f59e0b',       // Amber - middle
  bottom3: '#ef4444',      // Red - losing
  reference: isDark ? '#475569' : '#cbd5e1',
});

/**
 * Custom tooltip for the bar chart - shows POINTS (not ranks)
 */
const CustomTooltip = ({ active, payload, label, isDark, totalTeams, categoryLabel }) => {
  if (!active || !payload || !payload.length) return null;

  const colors = getChartColors(isDark);
  const currentPoints = payload.find(p => p.dataKey === 'currentPoints')?.value;
  const projectedPoints = payload.find(p => p.dataKey === 'projectedPoints')?.value;

  // Convert points back to rank for display
  const currentRank = totalTeams - currentPoints + 1;
  const projectedRank = totalTeams - projectedPoints + 1;

  const pointsDiff = projectedPoints - currentPoints;
  const improvementText = pointsDiff > 0
    ? `+${pointsDiff} pts (improving)`
    : pointsDiff < 0
      ? `${pointsDiff} pts (declining)`
      : 'No change';
  const improvementColor = pointsDiff > 0
    ? colors.improving
    : pointsDiff < 0
      ? colors.declining
      : colors.neutral;

  const formatOrdinal = (rank) => {
    if (typeof rank !== 'number') return rank;
    const suffixes = ['th', 'st', 'nd', 'rd'];
    const v = rank % 100;
    return rank + (suffixes[(v - 20) % 10] || suffixes[v] || suffixes[0]);
  };

  return (
    <div
      style={{
        backgroundColor: colors.tooltip.bg,
        border: `1px solid ${colors.tooltip.border}`,
        borderRadius: '8px',
        padding: '12px 16px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        minWidth: '200px',
      }}
    >
      <p style={{
        color: colors.tooltip.text,
        fontWeight: 600,
        marginBottom: '8px',
        fontSize: '0.95rem',
      }}>
        {categoryLabel || label}
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: colors.current, fontSize: '0.85rem', fontWeight: 500 }}>
            Current:
          </span>
          <span style={{ color: colors.tooltip.text, fontSize: '0.85rem', fontWeight: 600 }}>
            {currentPoints} pts ({formatOrdinal(currentRank)})
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: colors.projected, fontSize: '0.85rem', fontWeight: 500 }}>
            Projected:
          </span>
          <span style={{ color: colors.tooltip.text, fontSize: '0.85rem', fontWeight: 600 }}>
            {projectedPoints} pts ({formatOrdinal(projectedRank)})
          </span>
        </div>
        <div style={{
          marginTop: '4px',
          paddingTop: '8px',
          borderTop: `1px solid ${colors.tooltip.border}`,
          textAlign: 'center',
        }}>
          <span style={{ color: improvementColor, fontSize: '0.8rem', fontWeight: 600 }}>
            {improvementText}
          </span>
        </div>
      </div>
    </div>
  );
};

/**
 * Team selector dropdown component
 */
const TeamSelector = ({ teams, selectedTeamId, onTeamChange, isDark }) => {
  const colors = getChartColors(isDark);

  return (
    <div className="team-selector">
      <label
        htmlFor="team-select"
        style={{
          color: colors.axis,
          marginRight: '12px',
          fontSize: '0.9rem',
          fontWeight: 500,
        }}
      >
        View Team:
      </label>
      <select
        id="team-select"
        value={selectedTeamId || ''}
        onChange={(e) => onTeamChange(e.target.value)}
        className="team-select-dropdown"
        style={{
          backgroundColor: isDark ? '#1e293b' : '#ffffff',
          color: isDark ? '#f8fafc' : '#0f172a',
          border: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`,
          borderRadius: '6px',
          padding: '8px 32px 8px 12px',
          fontSize: '0.9rem',
          cursor: 'pointer',
          outline: 'none',
          minWidth: '200px',
        }}
      >
        {teams.map((team) => (
          <option
            key={team.team_id || team.espn_team_id || team.id}
            value={team.team_id || team.espn_team_id || team.id}
          >
            {team.team_name || team.name}
            {team.is_user_team ? ' (You)' : ''}
          </option>
        ))}
      </select>
    </div>
  );
};

/**
 * CategoryComparisonChart - Grouped bar chart for Roto league category comparison
 * Shows current vs projected rank for each category for a selected team
 */
function CategoryComparisonChart({
  teams = [],
  currentStandings = [],
  projectedStandings = [],
  categoryData = {},
  scoringCategories = null,
  userTeamId = null,
  title = 'Category Comparison: Current vs Projected',
  startLimitsEnabled = false,
}) {
  const { isDark } = useTheme();
  const colors = getChartColors(isDark);

  // State for selected team (default to user's team)
  const [selectedTeamId, setSelectedTeamId] = useState(
    userTeamId || teams[0]?.team_id || teams[0]?.espn_team_id || teams[0]?.id
  );

  const totalTeams = teams.length || currentStandings.length || 10;

  // Memoize categories to prevent recreating the array on every render
  const categories = useMemo(() => {
    return scoringCategories || [];
  }, [scoringCategories]);

  // Build chart data for the selected team - converts RANKS to POINTS
  // Points = totalTeams - rank + 1 (e.g., 10-team league: 1st place = 10 pts, 10th = 1 pt)
  const chartData = useMemo(() => {
    const selectedTeamCurrent = currentStandings.find(
      t => (t.team_id || t.espn_team_id || t.id) === selectedTeamId ||
           String(t.team_id || t.espn_team_id || t.id) === String(selectedTeamId)
    );
    const selectedTeamProjected = projectedStandings.find(
      t => (t.team_id || t.espn_team_id || t.id) === selectedTeamId ||
           String(t.team_id || t.espn_team_id || t.id) === String(selectedTeamId)
    );

    return categories.map(cat => {
      // Get current rank from standings
      let currentRank = selectedTeamCurrent?.category_ranks?.[cat.key]
        || selectedTeamCurrent?.categories?.[cat.key]
        || null;

      // Get projected rank
      let projectedRank = selectedTeamProjected?.projected_category_ranks?.[cat.key]
        || selectedTeamProjected?.category_ranks?.[cat.key]
        || selectedTeamProjected?.categories?.[cat.key]
        || currentRank;

      // Try to get from categoryData if available
      if (categoryData?.[cat.key]?.teams) {
        const teamCatData = categoryData[cat.key].teams.find(
          t => String(t.team_id) === String(selectedTeamId)
        );
        if (teamCatData) {
          currentRank = teamCatData.rank || currentRank;
        }
      }

      // Default to middle of pack if no data
      if (!currentRank) currentRank = Math.ceil(totalTeams / 2);
      if (!projectedRank) projectedRank = currentRank;

      // Convert ranks to POINTS (higher = better)
      // 1st place = totalTeams pts, last place = 1 pt
      const currentPoints = totalTeams - currentRank + 1;
      const projectedPoints = totalTeams - projectedRank + 1;
      const pointsDiff = projectedPoints - currentPoints;

      return {
        category: cat.abbr,
        categoryKey: cat.key,
        categoryLabel: cat.label,
        currentRank,
        projectedRank,
        currentPoints,
        projectedPoints,
        pointsDiff,
        isTop3Current: currentRank <= 3,
        isTop3Projected: projectedRank <= 3,
        isBottom3Current: currentRank > totalTeams - 3,
        isBottom3Projected: projectedRank > totalTeams - 3,
      };
    });
  }, [selectedTeamId, currentStandings, projectedStandings, categoryData, categories, totalTeams]);

  // Get selected team name for display
  const selectedTeamName = useMemo(() => {
    const team = [...teams, ...currentStandings].find(
      t => String(t.team_id || t.espn_team_id || t.id) === String(selectedTeamId)
    );
    return team?.team_name || team?.name || 'Team';
  }, [selectedTeamId, teams, currentStandings]);

  // Calculate summary stats
  const summaryStats = useMemo(() => {
    const improving = chartData.filter(d => d.pointsDiff > 0).length;
    const declining = chartData.filter(d => d.pointsDiff < 0).length;
    const noChange = chartData.filter(d => d.pointsDiff === 0).length;
    const top3Current = chartData.filter(d => d.isTop3Current).length;
    const top3Projected = chartData.filter(d => d.isTop3Projected).length;

    // Calculate total points change
    const totalPointsChange = chartData.reduce((sum, d) => sum + d.pointsDiff, 0);

    return { improving, declining, noChange, top3Current, top3Projected, totalPointsChange };
  }, [chartData]);

  // All teams for selector (combine from different sources)
  const allTeams = useMemo(() => {
    const teamMap = new Map();

    [...teams, ...currentStandings, ...projectedStandings].forEach(team => {
      const id = team.team_id || team.espn_team_id || team.id;
      if (id && !teamMap.has(String(id))) {
        teamMap.set(String(id), {
          team_id: id,
          team_name: team.team_name || team.name,
          is_user_team: team.is_user_team || id === userTeamId,
        });
      }
    });

    return Array.from(teamMap.values());
  }, [teams, currentStandings, projectedStandings, userTeamId]);

  // Custom bar colors based on points (higher = better)
  const getCurrentBarColor = (entry) => {
    if (entry.isTop3Current) return colors.top3;
    if (entry.isBottom3Current) return colors.bottom3;
    return colors.current;
  };

  const getProjectedBarColor = (entry) => {
    if (entry.pointsDiff > 0) return colors.improving;
    if (entry.pointsDiff < 0) return colors.declining;
    return colors.projected;
  };

  // Midpoint for reference line (average points)
  const midpoint = (totalTeams + 1) / 2; // 5.5 for 10-team league

  if (!allTeams.length) {
    return (
      <div className="category-comparison-chart">
        <h3 className="chart-title">{title}</h3>
        <div className="chart-empty">No team data available</div>
      </div>
    );
  }

  if (!categories.length) {
    return (
      <div className="category-comparison-chart">
        <h3 className="chart-title">{title}</h3>
        <div className="chart-empty">No category data available. Please refresh league data.</div>
      </div>
    );
  }

  return (
    <div className="category-comparison-chart">
      <div className="chart-header">
        <div className="chart-title-wrapper">
          <h3 className="chart-title">{title}</h3>
          {startLimitsEnabled && (
            <span
              className="start-limits-indicator"
              title="Projected ranks account for position start limits using day-by-day simulation"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                marginLeft: '8px',
                padding: '2px 8px',
                backgroundColor: isDark ? 'rgba(16, 185, 129, 0.15)' : 'rgba(16, 185, 129, 0.1)',
                borderRadius: '4px',
                fontSize: '0.7rem',
                color: '#10b981',
                fontWeight: 500,
                cursor: 'help',
              }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="16" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12.01" y2="8" />
              </svg>
              Start Limit Adjusted
            </span>
          )}
        </div>
        <TeamSelector
          teams={allTeams}
          selectedTeamId={selectedTeamId}
          onTeamChange={setSelectedTeamId}
          isDark={isDark}
        />
      </div>

      {/* Summary Stats */}
      <div className="chart-summary" style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '24px',
        marginBottom: '16px',
        padding: '12px',
        backgroundColor: isDark ? 'rgba(30, 41, 59, 0.5)' : 'rgba(241, 245, 249, 0.8)',
        borderRadius: '8px',
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            color: colors.improving
          }}>
            {summaryStats.improving}
          </div>
          <div style={{ fontSize: '0.75rem', color: colors.axis }}>Improving</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            color: colors.neutral
          }}>
            {summaryStats.noChange}
          </div>
          <div style={{ fontSize: '0.75rem', color: colors.axis }}>No Change</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            color: colors.declining
          }}>
            {summaryStats.declining}
          </div>
          <div style={{ fontSize: '0.75rem', color: colors.axis }}>Declining</div>
        </div>
        <div style={{
          textAlign: 'center',
          paddingLeft: '24px',
          borderLeft: `1px solid ${colors.grid}`,
        }}>
          <div style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            color: colors.top3
          }}>
            {summaryStats.top3Current} → {summaryStats.top3Projected}
          </div>
          <div style={{ fontSize: '0.75rem', color: colors.axis }}>Top 3 Categories</div>
        </div>
      </div>

      {/* Team name subtitle */}
      <div style={{
        textAlign: 'center',
        marginBottom: '8px',
        color: colors.axis,
        fontSize: '0.85rem',
      }}>
        Showing category ranks for <strong style={{ color: isDark ? '#f8fafc' : '#0f172a' }}>{selectedTeamName}</strong>
      </div>

      {/* Chart - showing POINTS (higher = better) */}
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 10, bottom: 60 }}
            barCategoryGap="20%"
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={colors.grid}
              vertical={false}
            />
            <XAxis
              dataKey="category"
              stroke={colors.axis}
              tick={{ fill: colors.axis, fontSize: 11, fontWeight: 500 }}
              axisLine={{ stroke: colors.grid }}
              tickLine={{ stroke: colors.grid }}
              interval={0}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis
              domain={[0, totalTeams]}
              stroke={colors.axis}
              tick={{ fill: colors.axis, fontSize: 11 }}
              axisLine={{ stroke: colors.grid }}
              tickLine={{ stroke: colors.grid }}
              label={{
                value: 'Points',
                angle: -90,
                position: 'insideLeft',
                fill: colors.axis,
                fontSize: 11,
                offset: 10,
              }}
              ticks={[0, 2, 4, 6, 8, 10]}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                const entry = chartData.find(d => d.category === label);
                return (
                  <CustomTooltip
                    active={active}
                    payload={payload}
                    label={label}
                    isDark={isDark}
                    totalTeams={totalTeams}
                    categoryLabel={entry?.categoryLabel}
                  />
                );
              }}
              cursor={{ fill: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' }}
            />

            {/* Reference line at midpoint (average) */}
            <ReferenceLine
              y={midpoint}
              stroke={colors.middle}
              strokeDasharray="5 5"
              label={{
                value: `Avg (${midpoint} pts)`,
                fill: colors.middle,
                fontSize: 10,
                position: 'right',
              }}
            />

            <Bar
              dataKey="currentPoints"
              name="Current Points"
              radius={[4, 4, 0, 0]}
              maxBarSize={35}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`current-${index}`}
                  fill={getCurrentBarColor(entry)}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
            <Bar
              dataKey="projectedPoints"
              name="Projected Points"
              radius={[4, 4, 0, 0]}
              maxBarSize={35}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`projected-${index}`}
                  fill={getProjectedBarColor(entry)}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
            <Legend
              wrapperStyle={{ paddingTop: '10px' }}
              formatter={(value) => (
                <span style={{ color: colors.axis, fontSize: '0.8rem' }}>{value}</span>
              )}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Legend for colors */}
      <div className="chart-legend-custom" style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '20px',
        marginTop: '12px',
        flexWrap: 'wrap',
      }}>
        <span className="legend-item" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            width: '12px',
            height: '12px',
            borderRadius: '2px',
            backgroundColor: colors.top3
          }}></span>
          <span style={{ color: colors.axis, fontSize: '0.75rem' }}>Top 3 ({totalTeams}-{totalTeams-2} pts)</span>
        </span>
        <span className="legend-item" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            width: '12px',
            height: '12px',
            borderRadius: '2px',
            backgroundColor: colors.current
          }}></span>
          <span style={{ color: colors.axis, fontSize: '0.75rem' }}>Middle</span>
        </span>
        <span className="legend-item" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            width: '12px',
            height: '12px',
            borderRadius: '2px',
            backgroundColor: colors.bottom3
          }}></span>
          <span style={{ color: colors.axis, fontSize: '0.75rem' }}>Bottom 3 (1-3 pts)</span>
        </span>
        <span style={{
          borderLeft: `1px solid ${colors.grid}`,
          paddingLeft: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}>
          <span style={{
            width: '12px',
            height: '12px',
            borderRadius: '2px',
            backgroundColor: colors.improving
          }}></span>
          <span style={{ color: colors.axis, fontSize: '0.75rem' }}>Gaining Points</span>
        </span>
        <span className="legend-item" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            width: '12px',
            height: '12px',
            borderRadius: '2px',
            backgroundColor: colors.declining
          }}></span>
          <span style={{ color: colors.axis, fontSize: '0.75rem' }}>Losing Points</span>
        </span>
      </div>

      <p className="chart-subtitle" style={{
        textAlign: 'center',
        color: colors.axis,
        fontSize: '0.75rem',
        marginTop: '12px',
      }}>
        Higher points = better (1st place = {totalTeams} pts, last place = 1 pt). Bars above the average line ({midpoint} pts) indicate category strength.
      </p>
    </div>
  );
}

CategoryComparisonChart.propTypes = {
  teams: PropTypes.array,
  currentStandings: PropTypes.arrayOf(
    PropTypes.shape({
      team_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      espn_team_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      team_name: PropTypes.string,
      name: PropTypes.string,
      category_ranks: PropTypes.object,
      categories: PropTypes.object,
      is_user_team: PropTypes.bool,
    })
  ),
  projectedStandings: PropTypes.arrayOf(
    PropTypes.shape({
      team_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      espn_team_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      team_name: PropTypes.string,
      name: PropTypes.string,
      projected_category_ranks: PropTypes.object,
      category_ranks: PropTypes.object,
      categories: PropTypes.object,
    })
  ),
  categoryData: PropTypes.object,
  scoringCategories: PropTypes.arrayOf(
    PropTypes.shape({
      key: PropTypes.string.isRequired,
      abbr: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
    })
  ),
  userTeamId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  title: PropTypes.string,
  startLimitsEnabled: PropTypes.bool,
};

export default CategoryComparisonChart;

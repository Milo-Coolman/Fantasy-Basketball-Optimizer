import React from 'react';
import PropTypes from 'prop-types';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  Legend,
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
  },
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  success: '#10b981',
  error: '#ef4444',
  warning: '#f59e0b',
  teamColors: [
    '#3b82f6', // blue
    '#10b981', // green
    '#f59e0b', // amber
    '#ef4444', // red
    '#8b5cf6', // purple
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#f97316', // orange
    '#84cc16', // lime
    '#6366f1', // indigo
  ],
});

/**
 * Custom tooltip component for charts
 */
const CustomTooltip = ({ active, payload, label, isDark, formatter }) => {
  if (!active || !payload || !payload.length) return null;

  const colors = getChartColors(isDark);

  return (
    <div
      style={{
        backgroundColor: colors.tooltip.bg,
        border: `1px solid ${colors.tooltip.border}`,
        borderRadius: '8px',
        padding: '12px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
      }}
    >
      <p style={{ color: colors.tooltip.text, fontWeight: 600, marginBottom: '8px' }}>
        {label}
      </p>
      {payload.map((entry, index) => (
        <p
          key={index}
          style={{
            color: entry.color || colors.tooltip.text,
            fontSize: '0.875rem',
            margin: '4px 0',
          }}
        >
          {entry.name}: {formatter ? formatter(entry.value) : entry.value}
        </p>
      ))}
    </div>
  );
};

/**
 * H2H Playoff Probability Chart
 * Shows probability trends over the season
 */
function H2HPlayoffChart({ data, teams, userTeamId, title }) {
  const { isDark } = useTheme();
  const colors = getChartColors(isDark);

  // If no data provided, generate sample data
  const chartData = data || generateSampleH2HData(teams);

  return (
    <div className="projections-chart">
      {title && <h3 className="chart-title">{title}</h3>}
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
            <XAxis
              dataKey="week"
              stroke={colors.axis}
              tick={{ fill: colors.axis, fontSize: 12 }}
              axisLine={{ stroke: colors.grid }}
            />
            <YAxis
              domain={[0, 100]}
              stroke={colors.axis}
              tick={{ fill: colors.axis, fontSize: 12 }}
              axisLine={{ stroke: colors.grid }}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip
              content={({ active, payload, label }) => (
                <CustomTooltip
                  active={active}
                  payload={payload}
                  label={label}
                  isDark={isDark}
                  formatter={(value) => `${Math.round(value)}%`}
                />
              )}
            />
            <ReferenceLine
              y={50}
              stroke={colors.warning}
              strokeDasharray="5 5"
              label={{
                value: '50%',
                fill: colors.warning,
                fontSize: 12,
                position: 'right',
              }}
            />
            {teams?.map((team, idx) => (
              <Line
                key={team.team_id || team.id || idx}
                type="monotone"
                dataKey={team.team_name || team.name}
                stroke={colors.teamColors[idx % colors.teamColors.length]}
                strokeWidth={
                  (team.team_id || team.id) === userTeamId ? 3 : 1.5
                }
                dot={false}
                activeDot={{ r: 6, strokeWidth: 2 }}
              />
            ))}
            <Legend
              wrapperStyle={{ paddingTop: '20px' }}
              formatter={(value) => (
                <span style={{ color: colors.axis, fontSize: '0.75rem' }}>{value}</span>
              )}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-subtitle">
        Teams above the 50% line are more likely to make playoffs
      </p>
    </div>
  );
}

/**
 * Roto Category Points Chart
 * Shows current vs projected points for each category (grouped bar chart)
 * Points scale: In a 10-team league, 1st place = 10 pts, 10th place = 1 pt
 */
function RotoCategoryChart({ data, title, numTeams = 10 }) {
  const { isDark } = useTheme();
  const colors = getChartColors(isDark);

  // Default categories with POINTS (not ranks)
  // Points = numTeams - rank + 1 (e.g., 10-team league: 1st = 10 pts, 10th = 1 pt)
  const defaultCategories = [
    { category: 'FG%', currentPoints: 2, projectedPoints: 3 },
    { category: 'FT%', currentPoints: 9, projectedPoints: 8 },
    { category: '3PM', currentPoints: 8, projectedPoints: 9 },
    { category: 'PTS', currentPoints: 6, projectedPoints: 8 },
    { category: 'REB', currentPoints: 4, projectedPoints: 5 },
    { category: 'AST', currentPoints: 7, projectedPoints: 7 },
    { category: 'STL', currentPoints: 3, projectedPoints: 6 },
    { category: 'BLK', currentPoints: 5, projectedPoints: 4 },
  ];

  // Transform data if it uses old rank format (current/projected) to new points format
  const transformData = (inputData) => {
    if (!inputData || inputData.length === 0) return defaultCategories;

    // Check if data already has points format
    if (inputData[0].currentPoints !== undefined) {
      return inputData;
    }

    // Convert from ranks to points (rank 1 = 10 pts, rank 10 = 1 pt in 10-team league)
    return inputData.map(item => ({
      category: item.category,
      currentPoints: item.current !== undefined ? (numTeams - item.current + 1) : item.currentPoints,
      projectedPoints: item.projected !== undefined ? (numTeams - item.projected + 1) : item.projectedPoints,
    }));
  };

  const chartData = transformData(data);

  // Midpoint for reference line (average points)
  const midpoint = (numTeams + 1) / 2; // 5.5 for 10-team league

  return (
    <div className="projections-chart">
      {title && <h3 className="chart-title">{title}</h3>}
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 10, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} vertical={false} />
            <XAxis
              dataKey="category"
              stroke={colors.axis}
              tick={{ fill: colors.axis, fontSize: 12, fontWeight: 500 }}
              axisLine={{ stroke: colors.grid }}
              tickLine={{ stroke: colors.grid }}
              angle={-45}
              textAnchor="end"
              height={60}
              interval={0}
            />
            <YAxis
              domain={[0, numTeams]}
              stroke={colors.axis}
              tick={{ fill: colors.axis, fontSize: 12 }}
              axisLine={{ stroke: colors.grid }}
              tickLine={{ stroke: colors.grid }}
              label={{
                value: 'Points',
                angle: -90,
                position: 'insideLeft',
                fill: colors.axis,
                fontSize: 12,
              }}
              ticks={[0, 2, 4, 6, 8, 10]}
            />
            <Tooltip
              content={({ active, payload, label }) => (
                <CustomTooltip
                  active={active}
                  payload={payload}
                  label={label}
                  isDark={isDark}
                  formatter={(value) => `${value} pts`}
                />
              )}
            />
            {/* Reference line at midpoint (average) */}
            <ReferenceLine
              y={midpoint}
              stroke={colors.warning}
              strokeDasharray="5 5"
              label={{
                value: `Avg (${midpoint} pts)`,
                fill: colors.warning,
                fontSize: 11,
                position: 'right',
              }}
            />
            <Bar
              dataKey="currentPoints"
              name="Current Points"
              fill={colors.primary}
              radius={[4, 4, 0, 0]}
              barSize={28}
            >
              {chartData.map((entry, index) => (
                <Cell key={`current-${index}`} fill={colors.primary} fillOpacity={0.85} />
              ))}
            </Bar>
            <Bar
              dataKey="projectedPoints"
              name="Projected Points"
              fill={colors.success}
              radius={[4, 4, 0, 0]}
              barSize={28}
            >
              {chartData.map((entry, index) => {
                // Color projected bar based on change from current
                const diff = entry.projectedPoints - entry.currentPoints;
                let barColor = colors.success; // Default green for improvement
                if (diff < 0) barColor = colors.error; // Red for decline
                else if (diff === 0) barColor = colors.secondary; // Purple for no change
                return <Cell key={`projected-${index}`} fill={barColor} fillOpacity={0.85} />;
              })}
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
      <div className="chart-legend-custom">
        <span className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: colors.success }}></span>
          Improving
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: colors.error }}></span>
          Declining
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: colors.secondary }}></span>
          No Change
        </span>
      </div>
      <p className="chart-subtitle">
        Higher points = better (1st place = {numTeams} pts, last place = 1 pt)
      </p>
    </div>
  );
}

/**
 * Generate sample H2H data for demonstration
 */
function generateSampleH2HData(teams) {
  const weeks = [];
  const numWeeks = 20;

  for (let i = 1; i <= numWeeks; i++) {
    const weekData = { week: `Week ${i}` };

    teams?.forEach((team, idx) => {
      // Generate realistic-looking probability curves
      const baseProb = 30 + (idx * 8); // Different starting points
      const trend = (i / numWeeks) * 20 * (idx % 2 === 0 ? 1 : -1); // Some trending up, some down
      const noise = (Math.random() - 0.5) * 10;
      const probability = Math.max(5, Math.min(95, baseProb + trend + noise));
      weekData[team.team_name || team.name] = probability;
    });

    weeks.push(weekData);
  }

  return weeks;
}

/**
 * Main ProjectionsChart component
 * Renders appropriate chart based on league type
 */
function ProjectionsChart({
  leagueType = 'H2H',
  data = null,
  teams = [],
  userTeamId = null,
  categoryData = null,
  title = null,
  numTeams = 10,
}) {
  const isH2H = leagueType.toUpperCase().includes('H2H');

  if (isH2H) {
    return (
      <H2HPlayoffChart
        data={data}
        teams={teams}
        userTeamId={userTeamId}
        title={title || 'Playoff Probability Over Time'}
      />
    );
  }

  return (
    <RotoCategoryChart
      data={categoryData || data}
      title={title || 'Current vs Projected Category Points'}
      numTeams={numTeams}
    />
  );
}

ProjectionsChart.propTypes = {
  leagueType: PropTypes.string,
  data: PropTypes.array,
  teams: PropTypes.arrayOf(
    PropTypes.shape({
      team_id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      team_name: PropTypes.string,
      name: PropTypes.string,
    })
  ),
  userTeamId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  categoryData: PropTypes.arrayOf(
    PropTypes.shape({
      category: PropTypes.string.isRequired,
      // New points-based format
      currentPoints: PropTypes.number,
      projectedPoints: PropTypes.number,
      // Legacy rank-based format (will be converted to points)
      current: PropTypes.number,
      projected: PropTypes.number,
    })
  ),
  title: PropTypes.string,
  numTeams: PropTypes.number, // Number of teams in league (default 10)
};

// Export sub-components for direct use
ProjectionsChart.H2H = H2HPlayoffChart;
ProjectionsChart.Roto = RotoCategoryChart;

export default ProjectionsChart;

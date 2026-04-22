import axios from 'axios';

/**
 * Axios instance configured for the Fantasy Basketball Optimizer API
 */
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Send cookies with requests
});

/**
 * Request interceptor - add any auth headers or logging
 */
api.interceptors.request.use(
  (config) => {
    // You can add auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * Response interceptor - handle common errors
 */
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle 401 Unauthorized - redirect to login
    if (error.response?.status === 401) {
      // Only redirect if not already on auth pages
      const currentPath = window.location.pathname;
      if (!currentPath.includes('/login') && !currentPath.includes('/register')) {
        window.location.href = '/login';
      }
    }

    // Handle 500 Server Error
    if (error.response?.status === 500) {
      console.error('Server error:', error.response.data);
    }

    return Promise.reject(error);
  }
);

// =============================================================================
// League API Functions
// =============================================================================

/**
 * Get all leagues for the current user
 */
export const fetchLeagues = async () => {
  const response = await api.get('/leagues');
  return response.data;
};

/**
 * Get a specific league by ID
 */
export const fetchLeague = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}`);
  return response.data;
};

/**
 * Create a new league
 */
export const createLeague = async (leagueData) => {
  const response = await api.post('/leagues', leagueData);
  return response.data;
};

/**
 * Delete a league
 */
export const deleteLeague = async (leagueId) => {
  const response = await api.delete(`/leagues/${leagueId}`);
  return response.data;
};

// =============================================================================
// Dashboard API Functions
// =============================================================================

/**
 * Fetch full dashboard data for a league
 * Returns: current_standings, projected_standings, user_team, insights, last_updated
 */
export const fetchDashboard = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/dashboard`);
  return response.data;
};

/**
 * Fetch only standings data (lightweight)
 */
export const fetchStandings = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/dashboard/standings`);
  return response.data;
};

/**
 * Fetch only insights data (lightweight)
 */
export const fetchInsights = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/dashboard/insights`);
  return response.data;
};

/**
 * Trigger a data refresh from ESPN for a league
 */
export const refreshLeagueData = async (leagueId) => {
  const response = await api.post(`/leagues/${leagueId}/refresh`);
  return response.data;
};

// =============================================================================
// Projections API Functions
// =============================================================================

/**
 * Get player projections for a league
 */
export const fetchProjections = async (leagueId, options = {}) => {
  const params = new URLSearchParams();
  if (options.teamId) params.append('team_id', options.teamId);
  if (options.playerId) params.append('player_id', options.playerId);

  const response = await api.get(`/leagues/${leagueId}/projections?${params}`);
  return response.data;
};

/**
 * Get playoff probability history for H2H leagues
 */
export const fetchPlayoffProbabilityHistory = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/projections/playoff-history`);
  return response.data;
};

// =============================================================================
// Trade API Functions
// =============================================================================

/**
 * Analyze a potential trade
 */
export const analyzeTrade = async (leagueId, tradeData) => {
  const response = await api.post(`/leagues/${leagueId}/trades/analyze`, tradeData);
  return response.data;
};

/**
 * Get trade suggestions for a team
 *
 * @param {number} leagueId - League ID
 * @param {number} teamId - User's team ID
 * @param {Object} projectedCategoryRanks - Projected category ranks from dashboard data
 * @param {Object} allTeamsProjectedRanks - Optional: all teams' projected ranks
 * @param {Object} teamRosters - Optional: all teams' rosters with z-scores (from dashboard)
 * @param {Object} leagueAverages - Optional: league averages for z-score calculation
 * @param {number} maxSuggestions - Maximum suggestions to return (default 5)
 */
export const fetchTradeSuggestions = async (
  leagueId,
  teamId,
  projectedCategoryRanks = {},
  allTeamsProjectedRanks = {},
  teamRosters = null,
  leagueAverages = null,
  maxSuggestions = 5
) => {
  const requestData = {
    team_id: teamId,
    projected_category_ranks: projectedCategoryRanks,
    all_teams_projected_ranks: allTeamsProjectedRanks,
    max_suggestions: maxSuggestions,
  };

  // Pass rosters with z-scores if available (to avoid recalculation)
  if (teamRosters) {
    requestData.team_rosters = teamRosters;
  }

  // Pass league averages if available (for consistent z-score calculation)
  if (leagueAverages) {
    requestData.league_averages = leagueAverages;
  }

  const response = await api.post(`/leagues/${leagueId}/trades/suggestions`, requestData);
  return response.data;
};

/**
 * Get trade history for a league
 */
export const fetchTradeHistory = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/trades/history`);
  return response.data;
};

/**
 * Get trade settings for a league
 *
 * @param {number} leagueId - League ID
 * @returns {Object} Trade settings including trade_suggestion_mode
 */
export const fetchTradeSettings = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/trade-settings`);
  return response.data;
};

/**
 * Update trade settings for a league
 *
 * @param {number} leagueId - League ID
 * @param {Object} settings - Trade settings to update
 * @param {string} settings.trade_suggestion_mode - 'conservative', 'normal', or 'aggressive'
 */
export const updateTradeSettings = async (leagueId, settings) => {
  const response = await api.put(`/leagues/${leagueId}/trade-settings`, settings);
  return response.data;
};

// =============================================================================
// Waiver Wire API Functions
// =============================================================================

/**
 * Get waiver wire recommendations
 */
export const fetchWaiverRecommendations = async (leagueId, options = {}) => {
  const params = new URLSearchParams();
  if (options.teamId) params.append('team_id', options.teamId);
  if (options.limit) params.append('limit', options.limit);

  const response = await api.get(`/leagues/${leagueId}/waivers/recommendations?${params}`);
  return response.data;
};

/**
 * Analyze a specific waiver wire add/drop move
 *
 * @param {number} leagueId - League ID
 * @param {Object} data - Analysis request data
 * @param {Object} data.player_to_add - Free agent player data to add
 * @param {Object} data.player_to_drop - (optional) Player to drop from roster
 * @param {Array} data.current_roster - Current roster players
 * @param {Object} data.league_averages - League averages for z-score calculation
 * @returns {Object} WaiverAnalysis results
 */
export const analyzeWaiver = async (leagueId, data) => {
  const response = await api.post(`/leagues/${leagueId}/waivers/analyze`, data);
  return response.data;
};

/**
 * Get waiver wire pickup suggestions
 *
 * @param {number} leagueId - League ID
 * @param {Object} data - Suggestion request data
 * @param {number} data.team_id - User's team ID
 * @param {Array} data.current_roster - Current roster players
 * @param {Object} data.league_averages - League averages for z-score calculation
 * @param {Array} data.weak_categories - Categories to prioritize improving
 * @param {number} data.max_suggestions - Maximum suggestions to return
 * @returns {Object} Waiver suggestions and drop candidates
 */
export const fetchWaiverSuggestions = async (leagueId, data) => {
  const response = await api.post(`/leagues/${leagueId}/waivers/suggestions`, data);
  return response.data;
};

/**
 * Get streaming recommendations for H2H leagues
 */
export const fetchStreamingRecommendations = async (leagueId, teamId) => {
  const response = await api.get(`/leagues/${leagueId}/waivers/streaming?team_id=${teamId}`);
  return response.data;
};

// =============================================================================
// Settings API Functions
// =============================================================================

/**
 * Get projection settings for a league
 */
export const fetchProjectionSettings = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/projection-settings`);
  return response.data;
};

/**
 * Update projection settings for a league
 */
export const updateProjectionSettings = async (leagueId, settings) => {
  const response = await api.put(`/leagues/${leagueId}/projection-settings`, settings);
  return response.data;
};

// =============================================================================
// Season Recap API Functions
// =============================================================================

/**
 * Fetch season recap data for a league
 * Returns: final standings, MVP, category leaders, user team recap
 */
export const fetchSeasonRecap = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/season-recap`);
  return response.data;
};

// =============================================================================
// Season Management API Functions
// =============================================================================

/**
 * Check if a new season is available for a league
 */
export const checkNewSeason = async (leagueId) => {
  const response = await api.get(`/leagues/${leagueId}/check-new-season`);
  return response.data;
};

/**
 * Switch league to a new season
 */
export const switchSeason = async (leagueId, newSeason) => {
  const response = await api.post(`/leagues/${leagueId}/switch-season`, {
    new_season: newSeason
  });
  return response.data;
};

// =============================================================================
// Health Check
// =============================================================================

/**
 * Check API health status
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;

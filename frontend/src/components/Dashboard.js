import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

/**
 * Dashboard component - main view after login
 */
function Dashboard() {
  const { user } = useAuth();
  const [leagues, setLeagues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchLeagues();
  }, []);

  const fetchLeagues = async () => {
    try {
      const response = await api.get('/leagues');
      setLeagues(response.data.leagues || []);
    } catch (err) {
      setError('Failed to load leagues');
      console.error('Error fetching leagues:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="dashboard">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading your leagues...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Dashboard</h1>
        <p>Welcome back, {user?.email}</p>
      </div>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {/* Leagues Section */}
      <section className="dashboard-section">
        <div className="section-header">
          <h2>Your Leagues</h2>
          <Link to="/league/setup" className="btn btn-primary">
            + Add League
          </Link>
        </div>

        {leagues.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">🏀</div>
            <h3>No leagues yet</h3>
            <p>Connect your ESPN Fantasy Basketball league to get started</p>
            <Link to="/league/setup" className="btn btn-primary">
              Add Your First League
            </Link>
          </div>
        ) : (
          <div className="leagues-grid">
            {leagues.map((league) => (
              <div key={league.id} className="league-card">
                <div className="league-card-header">
                  <h3>{league.league_name}</h3>
                  <span className="league-type">{league.league_type}</span>
                </div>
                <div className="league-card-body">
                  <div className="league-info">
                    <span className="info-label">Season</span>
                    <span className="info-value">{league.season}</span>
                  </div>
                  <div className="league-info">
                    <span className="info-label">Teams</span>
                    <span className="info-value">{league.num_teams || '-'}</span>
                  </div>
                  <div className="league-info">
                    <span className="info-label">Last Updated</span>
                    <span className="info-value">
                      {league.last_updated
                        ? new Date(league.last_updated).toLocaleDateString()
                        : 'Never'}
                    </span>
                  </div>
                </div>
                <div className="league-card-footer">
                  <Link to={`/league/${league.id}`} className="btn btn-secondary btn-sm">
                    View Details
                  </Link>
                  <button className="btn btn-outline btn-sm">
                    Refresh Data
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Quick Actions */}
      {leagues.length > 0 && (
        <section className="dashboard-section">
          <h2>Quick Actions</h2>
          <div className="quick-actions">
            <div className="action-card">
              <span className="action-icon">📊</span>
              <h3>Projections</h3>
              <p>View season projections and standings</p>
            </div>
            <div className="action-card">
              <span className="action-icon">🔄</span>
              <h3>Trade Analyzer</h3>
              <p>Analyze potential trades</p>
            </div>
            <div className="action-card">
              <span className="action-icon">📋</span>
              <h3>Waiver Wire</h3>
              <p>Get pickup recommendations</p>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default Dashboard;

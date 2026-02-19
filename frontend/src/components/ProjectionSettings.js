import React, { useState, useEffect } from 'react';
import { fetchProjectionSettings, updateProjectionSettings } from '../services/api';

/**
 * ProjectionSettings - Configure game projection method for a league
 *
 * Options:
 * - Adaptive: Tiered game rates based on games played (default)
 * - Flat Rate: Fixed percentage for all players
 */
function ProjectionSettings({ leagueId, onSettingsChange }) {
  const [settings, setSettings] = useState({
    projection_method: 'adaptive',
    flat_game_rate: 0.85,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  // Load current settings on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const data = await fetchProjectionSettings(leagueId);
        setSettings({
          projection_method: data.projection_method || 'adaptive',
          flat_game_rate: data.flat_game_rate || 0.85,
        });
        setError('');
      } catch (err) {
        console.error('Error loading projection settings:', err);
        setError('Failed to load settings');
      } finally {
        setLoading(false);
      }
    };

    loadSettings();
  }, [leagueId]);

  /**
   * Handle method change (Adaptive or Flat Rate)
   */
  const handleMethodChange = (method) => {
    setSettings(prev => ({ ...prev, projection_method: method }));
    setSuccess('');
  };

  /**
   * Handle flat rate slider change
   */
  const handleRateChange = (e) => {
    const value = parseFloat(e.target.value);
    setSettings(prev => ({ ...prev, flat_game_rate: value }));
    setSuccess('');
  };

  /**
   * Save settings to backend
   */
  const handleSave = async () => {
    setSaving(true);
    setError('');
    setSuccess('');

    try {
      await updateProjectionSettings(leagueId, settings);
      setSuccess('Settings saved! Refresh projections to apply.');
      if (onSettingsChange) {
        onSettingsChange(settings);
      }
    } catch (err) {
      console.error('Error saving settings:', err);
      setError(err.response?.data?.error || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="projection-settings">
        <div className="settings-header" onClick={() => setIsExpanded(!isExpanded)}>
          <span className="settings-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
            </svg>
          </span>
          <span className="settings-title">Projection Settings</span>
          <span className="loading-text">Loading...</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`projection-settings ${isExpanded ? 'expanded' : ''}`}>
      <button
        className="settings-header"
        onClick={() => setIsExpanded(!isExpanded)}
        type="button"
      >
        <span className="settings-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
          </svg>
        </span>
        <span className="settings-title">Projection Settings</span>
        <span className={`chevron ${isExpanded ? 'expanded' : ''}`}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </span>
      </button>

      {isExpanded && (
        <div className="settings-content">
          {error && <div className="alert alert-error small">{error}</div>}
          {success && <div className="alert alert-success small">{success}</div>}

          <div className="settings-section">
            <h4>Game Projection Method</h4>
            <p className="section-description">
              Choose how remaining games are projected for each player
            </p>

            <div className="method-options">
              {/* Adaptive Option */}
              <label className={`method-option ${settings.projection_method === 'adaptive' ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="projection_method"
                  value="adaptive"
                  checked={settings.projection_method === 'adaptive'}
                  onChange={() => handleMethodChange('adaptive')}
                />
                <div className="option-content">
                  <span className="option-title">Adaptive (Recommended)</span>
                  <span className="option-description">
                    Adjusts rate based on games played:
                  </span>
                  <ul className="option-details">
                    <li>0-4 GP: 90% of remaining games (grace period)</li>
                    <li>5+ GP: Actual game rate (minimum 75%)</li>
                  </ul>
                </div>
              </label>

              {/* Flat Rate Option */}
              <label className={`method-option ${settings.projection_method === 'flat_rate' ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="projection_method"
                  value="flat_rate"
                  checked={settings.projection_method === 'flat_rate'}
                  onChange={() => handleMethodChange('flat_rate')}
                />
                <div className="option-content">
                  <span className="option-title">Flat Rate</span>
                  <span className="option-description">
                    Apply the same rate to all players regardless of games played
                  </span>
                </div>
              </label>
            </div>
          </div>

          {/* Flat Rate Slider - Only show when flat_rate is selected */}
          {settings.projection_method === 'flat_rate' && (
            <div className="settings-section rate-section">
              <h4>Game Rate Percentage</h4>
              <div className="rate-control">
                <input
                  type="range"
                  min="0.70"
                  max="1.00"
                  step="0.01"
                  value={settings.flat_game_rate}
                  onChange={handleRateChange}
                  className="rate-slider"
                />
                <div className="rate-value">
                  <span className="rate-number">{Math.round(settings.flat_game_rate * 100)}%</span>
                  <span className="rate-label">of remaining games</span>
                </div>
              </div>
              <p className="rate-hint">
                Higher rates assume players play most games. Lower rates account for rest days and minor injuries.
              </p>
            </div>
          )}

          <div className="settings-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? (
                <>
                  <span className="btn-spinner"></span>
                  Saving...
                </>
              ) : (
                'Save Settings'
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProjectionSettings;

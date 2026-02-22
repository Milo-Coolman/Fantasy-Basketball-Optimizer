import React, { useState, useEffect } from 'react';
import { fetchTradeSettings, updateTradeSettings } from '../services/api';

/**
 * TradeSettings - Configure trade suggestion aggressiveness for a league
 *
 * Options:
 * - Conservative: Only very fair trades (-0.5 to +0.5 z-score)
 * - Normal: Slightly favorable trades OK (-0.25 to +1.0 z-score)
 * - Aggressive: Only trades that benefit you (0.0 to +1.5 z-score)
 */
function TradeSettings({ leagueId, onSettingsChange }) {
  const [mode, setMode] = useState('normal');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  // Load current settings on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const data = await fetchTradeSettings(leagueId);
        setMode(data.trade_suggestion_mode || 'normal');
        setError('');
      } catch (err) {
        console.error('Error loading trade settings:', err);
        setError('Failed to load settings');
      } finally {
        setLoading(false);
      }
    };

    loadSettings();
  }, [leagueId]);

  /**
   * Handle mode change
   */
  const handleModeChange = (newMode) => {
    setMode(newMode);
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
      await updateTradeSettings(leagueId, { trade_suggestion_mode: mode });
      setSuccess('Settings saved! Trade suggestions will update on next refresh.');
      if (onSettingsChange) {
        onSettingsChange({ trade_suggestion_mode: mode });
      }
    } catch (err) {
      console.error('Error saving settings:', err);
      setError(err.response?.data?.error || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const modeOptions = [
    {
      value: 'conservative',
      title: 'Conservative',
      description: 'Only suggest very fair, balanced trades',
      details: 'Z-score range: -0.5 to +0.5',
      icon: '🛡️',
    },
    {
      value: 'normal',
      title: 'Normal (Recommended)',
      description: 'Suggest trades that are fair or slightly favor you',
      details: 'Z-score range: -0.25 to +1.0',
      icon: '⚖️',
    },
    {
      value: 'aggressive',
      title: 'Aggressive',
      description: 'Only suggest trades that clearly benefit you',
      details: 'Z-score range: 0.0 to +1.5',
      icon: '🎯',
    },
  ];

  if (loading) {
    return (
      <div className="trade-settings">
        <div className="settings-header" onClick={() => setIsExpanded(!isExpanded)}>
          <span className="settings-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5" />
            </svg>
          </span>
          <span className="settings-title">Trade Settings</span>
          <span className="loading-text">Loading...</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`trade-settings ${isExpanded ? 'expanded' : ''}`}>
      <button
        className="settings-header"
        onClick={() => setIsExpanded(!isExpanded)}
        type="button"
      >
        <span className="settings-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5" />
          </svg>
        </span>
        <span className="settings-title">Trade Settings</span>
        <span className="current-mode-badge">{mode}</span>
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
            <h4>Trade Suggestion Aggressiveness</h4>
            <p className="section-description">
              Control what types of trades are suggested based on z-score value
            </p>

            <div className="method-options">
              {modeOptions.map((option) => (
                <label
                  key={option.value}
                  className={`method-option ${mode === option.value ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="trade_suggestion_mode"
                    value={option.value}
                    checked={mode === option.value}
                    onChange={() => handleModeChange(option.value)}
                  />
                  <div className="option-content">
                    <span className="option-title">
                      <span className="option-icon">{option.icon}</span>
                      {option.title}
                    </span>
                    <span className="option-description">{option.description}</span>
                    <span className="option-details">{option.details}</span>
                  </div>
                </label>
              ))}
            </div>
          </div>

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

export default TradeSettings;

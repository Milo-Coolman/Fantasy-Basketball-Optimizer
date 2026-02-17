import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import api from '../services/api';

/**
 * LeagueSetup - Multi-step wizard for adding ESPN Fantasy Basketball leagues
 *
 * Steps:
 * 1. Enter ESPN league URL
 * 2. Instructions for finding ESPN cookies
 * 3. Enter ESPN_S2 and SWID cookies
 * 4. Verify and import league data
 */
function LeagueSetup() {
  const [step, setStep] = useState(1);
  const totalSteps = 4;

  const [formData, setFormData] = useState({
    leagueUrl: '',
    espnS2: '',
    swid: '',
  });

  const [parsedData, setParsedData] = useState(null);
  const [urlError, setUrlError] = useState('');
  const [cookieErrors, setCookieErrors] = useState({ espnS2: '', swid: '' });
  const [loading, setLoading] = useState(false);
  const [importStatus, setImportStatus] = useState(null);
  const [importResult, setImportResult] = useState(null);
  const [showChromeHelp, setShowChromeHelp] = useState(false);
  const [showFirefoxHelp, setShowFirefoxHelp] = useState(false);
  const [showSafariHelp, setShowSafariHelp] = useState(false);

  const navigate = useNavigate();

  // Step labels for progress indicator
  const stepLabels = [
    'League URL',
    'Instructions',
    'ESPN Cookies',
    'Import'
  ];

  /**
   * Parse ESPN league URL to extract league ID and season
   */
  const parseLeagueUrl = (url) => {
    if (!url) return null;

    try {
      // Handle various ESPN URL formats
      const urlObj = new URL(url);

      // Check if it's an ESPN fantasy URL
      if (!urlObj.hostname.includes('espn.com')) {
        return { error: 'Please enter a valid ESPN Fantasy URL' };
      }

      if (!urlObj.pathname.includes('basketball') && !urlObj.pathname.includes('fba')) {
        return { error: 'Please enter a Fantasy Basketball league URL' };
      }

      const params = new URLSearchParams(urlObj.search);
      const leagueId = params.get('leagueId');

      if (!leagueId) {
        return { error: 'Could not find league ID in URL. Make sure you copy the full URL from your league page.' };
      }

      // Get season - default to current season if not specified
      let season = params.get('seasonId');
      if (!season) {
        const now = new Date();
        // NBA season starts in October, so if we're before October, use current year
        // Otherwise use next year (e.g., 2024-25 season = 2025)
        season = now.getMonth() >= 9 ? now.getFullYear() + 1 : now.getFullYear();
      }

      return {
        leagueId: parseInt(leagueId, 10),
        season: parseInt(season, 10),
        isValid: true
      };
    } catch (err) {
      return { error: 'Invalid URL format. Please paste the complete URL from your browser.' };
    }
  };

  /**
   * Validate SWID cookie format
   */
  const validateSwid = (swid) => {
    if (!swid) return 'SWID is required';

    // SWID should be in format {GUID}
    const swidPattern = /^\{[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}\}$/i;

    if (!swid.startsWith('{') || !swid.endsWith('}')) {
      return 'SWID should be wrapped in curly braces { }';
    }

    if (!swidPattern.test(swid)) {
      return 'SWID format appears invalid. It should look like {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}';
    }

    return '';
  };

  /**
   * Validate ESPN_S2 cookie
   */
  const validateEspnS2 = (espnS2) => {
    if (!espnS2) return 'ESPN_S2 cookie is required';
    if (espnS2.length < 100) return 'ESPN_S2 cookie appears too short. Make sure you copied the entire value.';
    return '';
  };

  /**
   * Handle URL input change
   */
  const handleUrlChange = (e) => {
    const url = e.target.value;
    setFormData({ ...formData, leagueUrl: url });
    setUrlError('');

    if (url) {
      const parsed = parseLeagueUrl(url);
      if (parsed?.error) {
        setUrlError(parsed.error);
        setParsedData(null);
      } else {
        setParsedData(parsed);
      }
    } else {
      setParsedData(null);
    }
  };

  /**
   * Handle cookie input changes
   */
  const handleCookieChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value.trim() });

    // Clear error when user starts typing
    setCookieErrors({ ...cookieErrors, [name]: '' });
  };

  /**
   * Navigate to next step
   */
  const handleNextStep = () => {
    if (step === 1) {
      if (!parsedData || parsedData.error) {
        setUrlError('Please enter a valid ESPN Fantasy Basketball league URL');
        return;
      }
    }

    if (step === 3) {
      // Validate cookies before proceeding
      const espnS2Error = validateEspnS2(formData.espnS2);
      const swidError = validateSwid(formData.swid);

      if (espnS2Error || swidError) {
        setCookieErrors({ espnS2: espnS2Error, swid: swidError });
        return;
      }
    }

    if (step < totalSteps) {
      setStep(step + 1);
    }
  };

  /**
   * Navigate to previous step
   */
  const handlePrevStep = () => {
    if (step > 1) {
      setStep(step - 1);
      setImportStatus(null);
      setImportResult(null);
    }
  };

  /**
   * Submit and import the league
   */
  const handleImport = async () => {
    setLoading(true);
    setImportStatus('connecting');
    setImportResult(null);

    try {
      // Simulate progress steps for better UX
      await new Promise(resolve => setTimeout(resolve, 500));
      setImportStatus('validating');

      const response = await api.post('/leagues', {
        espn_league_id: parsedData.leagueId,
        season: parsedData.season,
        espn_s2: formData.espnS2,
        swid: formData.swid,
      });

      setImportStatus('success');
      setImportResult({
        success: true,
        leagueName: response.data.league_name,
        leagueId: response.data.league_id,
        teamsCount: response.data.sync_results?.teams_synced || 0,
        playersCount: response.data.sync_results?.players_synced || 0,
      });

    } catch (err) {
      setImportStatus('error');

      const errorMessage = err.response?.data?.error || 'Failed to import league';
      const statusCode = err.response?.status;

      let userFriendlyError = errorMessage;

      if (statusCode === 401) {
        userFriendlyError = 'ESPN authentication failed. Please verify your cookies are correct and not expired. Try logging out of ESPN and back in, then get fresh cookies.';
      } else if (statusCode === 404) {
        userFriendlyError = 'League not found on ESPN. Please check the league ID and season.';
      } else if (statusCode === 409) {
        userFriendlyError = 'This league has already been added to your account.';
      } else if (statusCode === 503) {
        userFriendlyError = 'Could not connect to ESPN. Please try again in a few minutes.';
      }

      setImportResult({
        success: false,
        error: userFriendlyError
      });
    } finally {
      setLoading(false);
    }
  };

  /**
   * Start import when reaching step 4
   */
  useEffect(() => {
    if (step === 4 && !importStatus) {
      handleImport();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  /**
   * Render progress indicator
   */
  const renderProgressIndicator = () => (
    <div className="setup-progress">
      {stepLabels.map((label, index) => (
        <React.Fragment key={index}>
          <div className={`progress-step ${step > index ? 'completed' : ''} ${step === index + 1 ? 'active' : ''}`}>
            <span className="step-number">
              {step > index + 1 ? (
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
                </svg>
              ) : (
                index + 1
              )}
            </span>
            <span className="step-label">{label}</span>
          </div>
          {index < stepLabels.length - 1 && (
            <div className={`progress-line ${step > index + 1 ? 'completed' : ''}`}></div>
          )}
        </React.Fragment>
      ))}
    </div>
  );

  /**
   * Step 1: League URL Input
   */
  const renderStep1 = () => (
    <div className="setup-step">
      <div className="step-header">
        <h2>Enter Your League URL</h2>
        <p>Copy and paste the URL from your ESPN Fantasy Basketball league page</p>
      </div>

      <div className="form-group">
        <label htmlFor="leagueUrl">ESPN League URL</label>
        <input
          type="url"
          id="leagueUrl"
          name="leagueUrl"
          value={formData.leagueUrl}
          onChange={handleUrlChange}
          placeholder="https://fantasy.espn.com/basketball/league?leagueId=12345"
          className={urlError ? 'input-error' : parsedData ? 'input-success' : ''}
          autoFocus
        />
        {urlError && <span className="form-error">{urlError}</span>}
        <small className="form-hint">
          Go to your league on ESPN, then copy the URL from your browser's address bar
        </small>
      </div>

      {parsedData && !parsedData.error && (
        <div className="parsed-info success">
          <div className="parsed-info-icon">
            <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
            </svg>
          </div>
          <div className="parsed-info-content">
            <p><strong>League ID:</strong> {parsedData.leagueId}</p>
            <p><strong>Season:</strong> {parsedData.season - 1}-{String(parsedData.season).slice(2)}</p>
          </div>
        </div>
      )}

      <div className="help-section">
        <h4>Where do I find my league URL?</h4>
        <ol>
          <li>Go to <a href="https://fantasy.espn.com/basketball" target="_blank" rel="noopener noreferrer">ESPN Fantasy Basketball</a></li>
          <li>Click on your league name</li>
          <li>Copy the entire URL from your browser's address bar</li>
        </ol>
      </div>

      <div className="setup-actions">
        <Link to="/dashboard" className="btn btn-secondary">
          Cancel
        </Link>
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleNextStep}
          disabled={!parsedData || parsedData.error}
        >
          Next Step
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" />
          </svg>
        </button>
      </div>
    </div>
  );

  /**
   * Step 2: Cookie Instructions
   */
  const renderStep2 = () => (
    <div className="setup-step">
      <div className="step-header">
        <h2>Getting Your ESPN Cookies</h2>
        <p>ESPN requires authentication cookies to access your private league data</p>
      </div>

      <div className="info-box">
        <div className="info-box-icon">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
          </svg>
        </div>
        <div className="info-box-content">
          <strong>Why do we need cookies?</strong>
          <p>ESPN uses cookies to authenticate users. We need your <code>espn_s2</code> and <code>SWID</code> cookies to access your league data on your behalf. These are stored securely and only used to fetch your league information.</p>
        </div>
      </div>

      <div className="browser-instructions">
        {/* Chrome Instructions */}
        <div className="browser-section">
          <button
            type="button"
            className={`browser-header ${showChromeHelp ? 'expanded' : ''}`}
            onClick={() => setShowChromeHelp(!showChromeHelp)}
          >
            <span className="browser-icon">
              <svg viewBox="0 0 24 24" width="24" height="24">
                <circle cx="12" cy="12" r="10" fill="#4285F4" />
                <circle cx="12" cy="12" r="4" fill="white" />
                <path d="M12 2v6.5" stroke="#EA4335" strokeWidth="4" />
                <path d="M12 2a10 10 0 0 1 8.66 5H12" fill="#EA4335" />
                <path d="M20.66 7a10 10 0 0 1-3.66 12.66L12 12" fill="#FBBC05" />
                <path d="M17 19.66A10 10 0 0 1 3.34 7L12 12" fill="#34A853" />
              </svg>
            </span>
            <span>Google Chrome</span>
            <svg className="chevron" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z" />
            </svg>
          </button>
          {showChromeHelp && (
            <div className="browser-content">
              <ol>
                <li>Go to <a href="https://www.espn.com" target="_blank" rel="noopener noreferrer">espn.com</a> and make sure you're logged in</li>
                <li>Press <kbd>F12</kbd> or right-click and select <strong>"Inspect"</strong></li>
                <li>Click the <strong>"Application"</strong> tab (you may need to click <code>&gt;&gt;</code> to find it)</li>
                <li>In the left sidebar, expand <strong>"Cookies"</strong> and click on <strong>"https://www.espn.com"</strong></li>
                <li>Find <code>espn_s2</code> in the list and double-click its value to select it, then copy</li>
                <li>Do the same for <code>SWID</code></li>
              </ol>
              <div className="screenshot-placeholder">
                <p>Tip: You can use the filter box to search for "espn_s2" or "SWID"</p>
              </div>
            </div>
          )}
        </div>

        {/* Firefox Instructions */}
        <div className="browser-section">
          <button
            type="button"
            className={`browser-header ${showFirefoxHelp ? 'expanded' : ''}`}
            onClick={() => setShowFirefoxHelp(!showFirefoxHelp)}
          >
            <span className="browser-icon">
              <svg viewBox="0 0 24 24" width="24" height="24">
                <circle cx="12" cy="12" r="10" fill="#FF7139" />
                <path d="M12 4c-4.4 0-8 3.6-8 8s3.6 8 8 8 8-3.6 8-8" fill="#FF7139" stroke="#FFD567" strokeWidth="2" />
              </svg>
            </span>
            <span>Mozilla Firefox</span>
            <svg className="chevron" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z" />
            </svg>
          </button>
          {showFirefoxHelp && (
            <div className="browser-content">
              <ol>
                <li>Go to <a href="https://www.espn.com" target="_blank" rel="noopener noreferrer">espn.com</a> and make sure you're logged in</li>
                <li>Press <kbd>F12</kbd> or right-click and select <strong>"Inspect"</strong></li>
                <li>Click the <strong>"Storage"</strong> tab</li>
                <li>In the left sidebar, expand <strong>"Cookies"</strong> and click on <strong>"https://www.espn.com"</strong></li>
                <li>Find and copy the values for <code>espn_s2</code> and <code>SWID</code></li>
              </ol>
            </div>
          )}
        </div>

        {/* Safari Instructions */}
        <div className="browser-section">
          <button
            type="button"
            className={`browser-header ${showSafariHelp ? 'expanded' : ''}`}
            onClick={() => setShowSafariHelp(!showSafariHelp)}
          >
            <span className="browser-icon">
              <svg viewBox="0 0 24 24" width="24" height="24">
                <circle cx="12" cy="12" r="10" fill="#006CFF" />
                <circle cx="12" cy="12" r="8" fill="white" />
                <path d="M12 4l2 8-8 2 2-8z" fill="#FF3B30" />
                <path d="M12 4l-2 8 8-2-2-8z" fill="#006CFF" />
              </svg>
            </span>
            <span>Safari</span>
            <svg className="chevron" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z" />
            </svg>
          </button>
          {showSafariHelp && (
            <div className="browser-content">
              <ol>
                <li>First, enable the Develop menu: Safari → Preferences → Advanced → Check "Show Develop menu in menu bar"</li>
                <li>Go to <a href="https://www.espn.com" target="_blank" rel="noopener noreferrer">espn.com</a> and make sure you're logged in</li>
                <li>Click <strong>Develop → Show Web Inspector</strong> (or press <kbd>Cmd+Option+I</kbd>)</li>
                <li>Click the <strong>"Storage"</strong> tab</li>
                <li>Expand <strong>"Cookies"</strong> in the left sidebar and click on espn.com</li>
                <li>Find and copy the values for <code>espn_s2</code> and <code>SWID</code></li>
              </ol>
            </div>
          )}
        </div>
      </div>

      <div className="warning-box">
        <div className="warning-box-icon">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
            <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
          </svg>
        </div>
        <div className="warning-box-content">
          <strong>Keep your cookies private!</strong>
          <p>Never share these cookie values with anyone. They provide access to your ESPN account.</p>
        </div>
      </div>

      <div className="setup-actions">
        <button type="button" className="btn btn-secondary" onClick={handlePrevStep}>
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
          </svg>
          Back
        </button>
        <button type="button" className="btn btn-primary" onClick={handleNextStep}>
          I Found My Cookies
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" />
          </svg>
        </button>
      </div>
    </div>
  );

  /**
   * Step 3: Cookie Input
   */
  const renderStep3 = () => (
    <div className="setup-step">
      <div className="step-header">
        <h2>Enter Your ESPN Cookies</h2>
        <p>Paste the cookie values you copied from your browser</p>
      </div>

      <div className="form-group">
        <label htmlFor="espnS2">
          ESPN_S2 Cookie
          <span className="required">*</span>
        </label>
        <textarea
          id="espnS2"
          name="espnS2"
          value={formData.espnS2}
          onChange={handleCookieChange}
          placeholder="Paste your espn_s2 cookie value here (it's a long string)"
          rows={4}
          className={cookieErrors.espnS2 ? 'input-error' : formData.espnS2 && !validateEspnS2(formData.espnS2) ? 'input-success' : ''}
        />
        {cookieErrors.espnS2 && <span className="form-error">{cookieErrors.espnS2}</span>}
        <small className="form-hint">
          This is a long encrypted string. Make sure you copy the entire value.
        </small>
      </div>

      <div className="form-group">
        <label htmlFor="swid">
          SWID Cookie
          <span className="required">*</span>
        </label>
        <input
          type="text"
          id="swid"
          name="swid"
          value={formData.swid}
          onChange={handleCookieChange}
          placeholder="{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}"
          className={cookieErrors.swid ? 'input-error' : formData.swid && !validateSwid(formData.swid) ? 'input-success' : ''}
        />
        {cookieErrors.swid && <span className="form-error">{cookieErrors.swid}</span>}
        <small className="form-hint">
          This should be a GUID wrapped in curly braces, like {'{12345678-1234-1234-1234-123456789ABC}'}
        </small>
      </div>

      <div className="review-section">
        <h4>Review Your League</h4>
        <div className="review-details">
          <div className="review-item">
            <span className="review-label">League ID</span>
            <span className="review-value">{parsedData?.leagueId}</span>
          </div>
          <div className="review-item">
            <span className="review-label">Season</span>
            <span className="review-value">{parsedData?.season - 1}-{String(parsedData?.season).slice(2)}</span>
          </div>
        </div>
      </div>

      <div className="setup-actions">
        <button type="button" className="btn btn-secondary" onClick={handlePrevStep}>
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
          </svg>
          Back
        </button>
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleNextStep}
          disabled={!formData.espnS2 || !formData.swid}
        >
          Import League
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" />
          </svg>
        </button>
      </div>
    </div>
  );

  /**
   * Step 4: Import Status
   */
  const renderStep4 = () => (
    <div className="setup-step import-step">
      <div className="step-header">
        <h2>
          {importStatus === 'success' ? 'League Imported!' :
           importStatus === 'error' ? 'Import Failed' :
           'Importing Your League'}
        </h2>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="import-loading">
          <div className="loading-spinner large"></div>
          <div className="import-status-text">
            {importStatus === 'connecting' && 'Connecting to ESPN...'}
            {importStatus === 'validating' && 'Validating credentials and fetching league data...'}
          </div>
          <div className="import-details">
            <p>League ID: {parsedData?.leagueId}</p>
            <p>Season: {parsedData?.season - 1}-{String(parsedData?.season).slice(2)}</p>
          </div>
        </div>
      )}

      {/* Success State */}
      {importStatus === 'success' && importResult?.success && (
        <div className="import-success">
          <div className="success-icon">
            <svg viewBox="0 0 24 24" width="64" height="64" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
            </svg>
          </div>
          <h3>{importResult.leagueName}</h3>
          <div className="import-stats">
            <div className="stat-item">
              <span className="stat-value">{importResult.teamsCount}</span>
              <span className="stat-label">Teams Synced</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{importResult.playersCount}</span>
              <span className="stat-label">Players Loaded</span>
            </div>
          </div>
          <p className="success-message">Your league has been successfully imported and is ready to use!</p>
          <div className="setup-actions">
            <button
              type="button"
              className="btn btn-primary btn-large"
              onClick={() => navigate('/dashboard')}
            >
              Go to Dashboard
              <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Error State */}
      {importStatus === 'error' && importResult && !importResult.success && (
        <div className="import-error">
          <div className="error-icon">
            <svg viewBox="0 0 24 24" width="64" height="64" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
            </svg>
          </div>
          <h3>Something went wrong</h3>
          <p className="error-message">{importResult.error}</p>

          <div className="error-help">
            <h4>Troubleshooting Tips:</h4>
            <ul>
              <li>Make sure you're logged into ESPN in your browser</li>
              <li>Try getting fresh cookies (log out and back in to ESPN)</li>
              <li>Ensure you copied the complete cookie values</li>
              <li>Verify the league URL is correct</li>
            </ul>
          </div>

          <div className="setup-actions">
            <button type="button" className="btn btn-secondary" onClick={() => setStep(3)}>
              <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
              </svg>
              Update Cookies
            </button>
            <button type="button" className="btn btn-primary" onClick={handleImport}>
              Try Again
            </button>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="league-setup">
      <div className="setup-card">
        <div className="setup-header">
          <h1>Add ESPN League</h1>
          <p>Connect your ESPN Fantasy Basketball league in a few easy steps</p>
        </div>

        {renderProgressIndicator()}

        <form onSubmit={(e) => e.preventDefault()}>
          {step === 1 && renderStep1()}
          {step === 2 && renderStep2()}
          {step === 3 && renderStep3()}
          {step === 4 && renderStep4()}
        </form>
      </div>
    </div>
  );
}

export default LeagueSetup;

import React from 'react';
import PropTypes from 'prop-types';

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
 * Waiver Wire Targets Section
 */
function WaiverTargets({ targets = [], maxItems = 3 }) {
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
      </div>
      <ul className="waiver-list">
        {targets.slice(0, maxItems).map((player, idx) => (
          <li key={player.id || idx} className="waiver-item">
            <div className="waiver-rank">{idx + 1}</div>
            <div className="waiver-info">
              <div className="waiver-player">
                <span className="player-name">{player.name}</span>
                <span className="player-meta">
                  {player.position} • {player.nba_team || player.team}
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
        ))}
      </ul>
    </div>
  );
}

/**
 * Trade Opportunities Section
 */
function TradeOpportunities({ opportunities = [], maxItems = 2 }) {
  if (!opportunities || opportunities.length === 0) {
    return (
      <div className="insight-section">
        <div className="insight-header">
          <TradeIcon />
          <h3>Trade Opportunities</h3>
        </div>
        <div className="insight-empty">
          <p>No trade opportunities identified</p>
          <span className="empty-hint">Your roster is well-balanced</span>
        </div>
      </div>
    );
  }

  return (
    <div className="insight-section">
      <div className="insight-header">
        <TradeIcon />
        <h3>Trade Opportunities</h3>
      </div>
      <ul className="trade-list">
        {opportunities.slice(0, maxItems).map((trade, idx) => (
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
            </div>
            {trade.value_gain && (
              <div className="trade-value">
                <span className="value-number">+{trade.value_gain}</span>
                <span className="value-label">Value</span>
              </div>
            )}
          </li>
        ))}
      </ul>
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
}) {
  return (
    <div className={`quick-insights ${compact ? 'compact' : ''} ${isRoto ? 'roto-mode' : ''}`}>
      {title && <h2 className="insights-title">{title}</h2>}

      <div className="insights-content">
        {/* Show category movements first for Roto leagues */}
        {(showMovements || isRoto) && categoryMovements.length > 0 && (
          <CategoryMovements movements={categoryMovements} maxItems={compact ? 3 : 4} />
        )}

        {showWaivers && (
          <WaiverTargets targets={waiverTargets} maxItems={compact ? 2 : 3} />
        )}

        {showTrades && (
          <TradeOpportunities opportunities={tradeOpportunities} maxItems={compact ? 1 : 2} />
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
};

// Export sub-components for direct use
QuickInsights.WaiverTargets = WaiverTargets;
QuickInsights.TradeOpportunities = TradeOpportunities;
QuickInsights.CategoryAnalysis = CategoryAnalysis;
QuickInsights.CategoryMovements = CategoryMovements;

export default QuickInsights;

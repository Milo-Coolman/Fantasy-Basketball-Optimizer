"""
Analysis module for Fantasy Basketball Optimizer.

Contains trade analysis, waiver analysis, roster optimization,
and recommendation engines.
"""

from backend.analysis.trade_analyzer import TradeAnalyzer, TradePlayer, TradeAnalysis
from backend.analysis.trade_suggestions import (
    TradeSuggestionGenerator,
    TradeSuggestion,
    generate_trade_suggestions,
)
from backend.analysis.waiver_analyzer import (
    WaiverAnalyzer,
    WaiverAnalysis,
    WaiverRecommendation,
    DropSuggestion,
)

__all__ = [
    # Trade analysis
    'TradeAnalyzer',
    'TradePlayer',
    'TradeAnalysis',
    'TradeSuggestionGenerator',
    'TradeSuggestion',
    'generate_trade_suggestions',
    # Waiver analysis
    'WaiverAnalyzer',
    'WaiverAnalysis',
    'WaiverRecommendation',
    'DropSuggestion',
]

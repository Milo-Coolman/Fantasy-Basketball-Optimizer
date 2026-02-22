"""
Analysis module for Fantasy Basketball Optimizer.

Contains trade analysis, roster optimization, and recommendation engines.
"""

from backend.analysis.trade_analyzer import TradeAnalyzer, TradePlayer, TradeAnalysis
from backend.analysis.trade_suggestions import (
    TradeSuggestionGenerator,
    TradeSuggestion,
    generate_trade_suggestions,
)

__all__ = [
    'TradeAnalyzer',
    'TradePlayer',
    'TradeAnalysis',
    'TradeSuggestionGenerator',
    'TradeSuggestion',
    'generate_trade_suggestions',
]

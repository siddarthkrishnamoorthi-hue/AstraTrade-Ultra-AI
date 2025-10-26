"""
Custom exceptions for the trading service
"""

class TradingServiceError(Exception):
    """Base exception for trading service errors"""
    pass

class ExecutorInitError(TradingServiceError):
    """Raised when executor initialization fails"""
    pass

class MarketDataError(TradingServiceError):
    """Raised when market data cannot be retrieved"""
    pass

class OrderExecutionError(TradingServiceError):
    """Raised when order execution fails"""
    pass

class ConnectionError(TradingServiceError):
    """Raised when connection to trading infrastructure fails"""
    pass
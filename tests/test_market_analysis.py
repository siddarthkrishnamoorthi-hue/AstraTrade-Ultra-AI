"""
Unit tests for Market Analysis components
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from advanced.market_analysis import MarketAnalyzer
from advanced.detectors import PatternDetector
from advanced.institutional_strategy import InstitutionalDetector

@pytest.fixture
def sample_data():
    """Create sample market data"""
    return pd.DataFrame({
        'open': np.random.random(100) * 100,
        'high': np.random.random(100) * 100,
        'low': np.random.random(100) * 100,
        'close': np.random.random(100) * 100,
        'volume': np.random.random(100) * 1000,
        'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='1H')
    })

class TestMarketAnalyzer:
    """Test suite for MarketAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create market analyzer instance"""
        return MarketAnalyzer()

    def test_volatility_calculation(self, analyzer, sample_data):
        """Test volatility calculation accuracy"""
        volatility = analyzer.calculate_volatility(sample_data)
        assert isinstance(volatility, float)
        assert 0 <= volatility <= 1

    @pytest.mark.benchmark
    def test_analysis_performance(self, analyzer, sample_data, benchmark):
        """Test analysis performance"""
        def analyze_iteration():
            return analyzer.analyze_market_state(sample_data)
        
        result = benchmark(analyze_iteration)
        assert result.stats.mean < 0.05  # Should complete within 50ms

    @pytest.mark.parametrize("timeframe", ['1H', '4H', '1D'])
    def test_multi_timeframe_analysis(self, analyzer, sample_data, timeframe):
        """Test analysis across different timeframes"""
        resampled_data = sample_data.resample(timeframe, on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        result = analyzer.analyze_market_state(resampled_data)
        assert isinstance(result, dict)
        assert 'volatility' in result
        assert 'trend_strength' in result

class TestPatternDetector:
    """Test suite for Pattern Detection"""

    @pytest.fixture
    def detector(self):
        """Create pattern detector instance"""
        return PatternDetector()

    def test_pattern_recognition(self, detector, sample_data):
        """Test pattern recognition accuracy"""
        patterns = detector.detect_patterns(sample_data)
        assert isinstance(patterns, list)
        
    @pytest.mark.parametrize("pattern_type", [
        'DOUBLE_TOP',
        'DOUBLE_BOTTOM',
        'HEAD_AND_SHOULDERS',
        'TRIANGLE'
    ])
    def test_specific_patterns(self, detector, pattern_type):
        """Test specific pattern detection"""
        # Create synthetic data for specific patterns
        data = self._create_pattern_data(pattern_type)
        patterns = detector.detect_patterns(data)
        assert pattern_type in [p['type'] for p in patterns]

    def _create_pattern_data(self, pattern_type):
        """Create synthetic data for pattern testing"""
        if pattern_type == 'DOUBLE_TOP':
            prices = [10, 12, 15, 12, 15, 12, 10]
        elif pattern_type == 'DOUBLE_BOTTOM':
            prices = [15, 12, 10, 12, 10, 12, 15]
        elif pattern_type == 'HEAD_AND_SHOULDERS':
            prices = [10, 15, 12, 17, 12, 15, 10]
        else:  # TRIANGLE
            prices = [10, 15, 11, 14, 12, 13, 12.5]
        
        return pd.DataFrame({
            'close': prices,
            'timestamp': pd.date_range(start='2025-01-01', periods=len(prices), freq='1H')
        })

class TestInstitutionalDetector:
    """Test suite for Institutional Trading Detection"""

    @pytest.fixture
    def detector(self):
        """Create institutional detector instance"""
        return InstitutionalDetector()

    def test_volume_analysis(self, detector, sample_data):
        """Test institutional volume analysis"""
        result = detector.analyze_volume_clusters(sample_data)
        assert isinstance(result, dict)
        assert 'significant_levels' in result
        
    def test_orderflow_detection(self, detector, sample_data):
        """Test orderflow analysis"""
        result = detector.detect_orderflow_patterns(sample_data)
        assert isinstance(result, list)
        assert all(isinstance(p, dict) for p in result)
        
    @pytest.mark.benchmark
    def test_detection_performance(self, detector, sample_data, benchmark):
        """Test detection performance"""
        def detection_iteration():
            return detector.analyze_volume_clusters(sample_data)
        
        result = benchmark(detection_iteration)
        assert result.stats.mean < 0.1  # Should complete within 100ms
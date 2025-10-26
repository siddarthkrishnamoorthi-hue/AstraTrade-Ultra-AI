"""
Symbol normalization and broker detection utilities
"""

import re
from typing import List, Dict, Optional

# Common broker-specific symbol patterns
SYMBOL_PATTERNS = {
    'suffix': r'^([A-Z]+)(c|m|v|\.pro|\.x|\.a|\.b|_cent|_standard)$',
    'prefix': r'^(pro\.|cent\.|std\.)([A-Z]+)$',
}

class SymbolManager:
    def __init__(self):
        self.symbol_cache: Dict[str, Dict] = {}
        self.broker_type: Optional[str] = None
    
    def detect_broker_type(self, sample_symbols: List[str]) -> str:
        """Detect broker type from symbol patterns"""
        patterns = {
            'ICMarkets': [r'\.pro$', r'\.x$'],
            'FXCM': [r'_cent$', r'_standard$'],
            'XM': [r'c$', r'm$'],
            'FBS': [r'\.a$', r'\.b$'],
            'Generic': [r'^[A-Z]+$']
        }
        
        for broker, pattern_list in patterns.items():
            matches = 0
            for pattern in pattern_list:
                for symbol in sample_symbols:
                    if re.search(pattern, symbol):
                        matches += 1
            if matches > len(sample_symbols) * 0.5:  # >50% match
                return broker
        return 'Generic'

    def normalize_symbol(self, symbol: str) -> str:
        """Convert broker-specific symbol to standard format"""
        # Check cache first
        if symbol in self.symbol_cache:
            return self.symbol_cache[symbol]['normalized']
        
        # Remove common suffixes and prefixes
        for pattern in SYMBOL_PATTERNS.values():
            match = re.match(pattern, symbol)
            if match:
                base_symbol = match.group(1)
                self.symbol_cache[symbol] = {
                    'normalized': base_symbol,
                    'original': symbol,
                    'suffix': match.group(2) if len(match.groups()) > 1 else None
                }
                return base_symbol
        
        # If no pattern matches, assume it's already normalized
        self.symbol_cache[symbol] = {
            'normalized': symbol,
            'original': symbol,
            'suffix': None
        }
        return symbol

    def denormalize_symbol(self, base_symbol: str, broker_type: Optional[str] = None) -> str:
        """Convert standard symbol to broker-specific format"""
        if not broker_type and not self.broker_type:
            return base_symbol
            
        broker = broker_type or self.broker_type
        suffix_map = {
            'ICMarkets': '.pro',
            'FXCM': '_standard',
            'XM': 'm',
            'FBS': '.a',
            'Generic': ''
        }
        
        return f"{base_symbol}{suffix_map.get(broker, '')}"

    def get_symbol_variations(self, base_symbol: str) -> List[str]:
        """Get all possible variations of a symbol"""
        variations = [base_symbol]  # Standard format
        
        # Add common variations
        suffixes = ['c', 'm', 'v', '.pro', '.x', '.a', '.b', '_cent', '_standard']
        variations.extend([f"{base_symbol}{suffix}" for suffix in suffixes])
        
        # Add prefix variations
        prefixes = ['pro.', 'cent.', 'std.']
        variations.extend([f"{prefix}{base_symbol}" for prefix in prefixes])
        
        return variations

    def get_tick_size(self, symbol: str) -> float:
        """Get symbol-specific tick size"""
        if 'XAU' in symbol or 'GOLD' in symbol:
            return 0.01  # Gold typically trades in 0.01 increments
        return 0.00001  # Standard forex pairs

    def get_contract_size(self, symbol: str) -> float:
        """Get symbol-specific contract size"""
        if 'XAU' in symbol or 'GOLD' in symbol:
            return 100  # Gold is typically 100 oz
        return 100000  # Standard forex lot size

    def get_margin_requirements(self, symbol: str, broker_type: Optional[str] = None) -> Dict:
        """Get margin requirements for symbol"""
        base_margin = 1000.0  # Default margin requirement
        
        # Adjust based on broker type
        broker_margins = {
            'ICMarkets': 0.8,
            'FXCM': 1.0,
            'XM': 1.2,
            'FBS': 0.9,
            'Generic': 1.0
        }
        
        margin_multiplier = broker_margins.get(broker_type or self.broker_type, 1.0)
        
        # Adjust for specific instruments
        if 'XAU' in symbol or 'GOLD' in symbol:
            base_margin *= 2  # Gold typically requires more margin
        
        return {
            'margin_required': base_margin * margin_multiplier,
            'maintenance_margin': base_margin * margin_multiplier * 0.7,
            'margin_call_level': 80.0  # Percentage
        }
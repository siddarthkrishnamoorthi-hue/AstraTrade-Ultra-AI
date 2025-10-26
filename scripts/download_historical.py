"""
Download and prepare historical data for model training
"""
import asyncio
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import argparse

class HistoricalDataManager:
    def __init__(self):
        self.logger = logging.getLogger("HistoricalData")
        self.data_dir = Path("data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def download_data(
        self,
        symbols: List[str],
        years: int = 3,
        timeframes: List[str] = None
    ):
        """Download historical data for multiple symbols"""
        timeframes = timeframes or ["M5", "M15", "H1", "H4"]
        
        # Initialize MT5
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            return False
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        for symbol in symbols:
            self.logger.info(f"Downloading data for {symbol}")
            
            for timeframe in timeframes:
                tf = self._get_mt5_timeframe(timeframe)
                
                # Download in chunks to handle large datasets
                current_start = start_date
                while current_start < end_date:
                    chunk_end = min(
                        current_start + timedelta(days=90),
                        end_date
                    )
                    
                    rates = mt5.copy_rates_range(
                        symbol,
                        tf,
                        current_start,
                        chunk_end
                    )
                    
                    if rates is None:
                        self.logger.error(
                            f"Failed to download {symbol} {timeframe}"
                        )
                        continue
                        
                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Save chunk
                    filename = self.data_dir / f"{symbol}_{timeframe}_{current_start.strftime('%Y%m%d')}.parquet"
                    df.to_parquet(filename)
                    
                    self.logger.info(
                        f"Saved {len(df)} rows for {symbol} {timeframe}"
                    )
                    
                    current_start = chunk_end
                    
                    # Sleep to avoid overloading
                    await asyncio.sleep(1)
                    
        mt5.shutdown()
        return True
        
    def prepare_training_data(
        self,
        symbols: List[str],
        timeframes: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Prepare data for model training"""
        timeframes = timeframes or ["M5", "M15", "H1", "H4"]
        results = {}
        
        for symbol in symbols:
            symbol_data = {}
            
            for timeframe in timeframes:
                # Load all chunks for this symbol/timeframe
                pattern = f"{symbol}_{timeframe}_*.parquet"
                files = list(self.data_dir.glob(pattern))
                
                if not files:
                    continue
                    
                # Concatenate chunks
                dfs = []
                for file in files:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                    
                if dfs:
                    # Combine and sort
                    combined = pd.concat(dfs)
                    combined = combined.sort_values('time')
                    combined = combined.drop_duplicates()
                    
                    # Add technical indicators
                    combined = self._add_technical_features(combined)
                    
                    symbol_data[timeframe] = combined
                    
            if symbol_data:
                results[symbol] = symbol_data
                
        return results
        
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        # Basic indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df
        
    def _get_mt5_timeframe(self, timeframe: str) -> int:
        """Convert string timeframe to MT5 constant"""
        return {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }[timeframe]

async def main():
    parser = argparse.ArgumentParser(description="Download historical data")
    parser.add_argument(
        '--pairs',
        nargs='+',
        default=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
        help='List of pairs to download'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='Years of historical data'
    )
    args = parser.parse_args()
    
    manager = HistoricalDataManager()
    await manager.download_data(args.pairs, args.years)

if __name__ == "__main__":
    asyncio.run(main())
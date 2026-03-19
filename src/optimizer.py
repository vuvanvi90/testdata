import os
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
from pathlib import Path
try:
    from tqdm import tqdm # Thư viện tạo thanh trình duyệt đẹp mắt
except ImportError:
    print("Vui lòng cài đặt tqdm: pip install tqdm")
    def tqdm(iterable, **kwargs): return iterable

from src.market_flow import MarketFlowAnalyzer 

class QuantOptimizer:
    """
    Động cơ Tối ưu hóa Trọng số bằng phương pháp Grid Search.
    Giả lập giao dịch quá khứ để tìm ra bộ cấu hình mang lại Max Sharpe Ratio.
    """
    def __init__(self, historical_signals=None, historical_prices=None):
        # historical_signals: DataFrame chứa các tín hiệu (POC, DTL...) đã được tính sẵn
        # historical_prices: DataFrame giá để tính lợi nhuận T+
        self.signals = historical_signals
        self.prices = historical_prices

        self.backtest_dir = Path('data/backtest')
        self.parquet_dir = Path('data/parquet')
        self._ensure_exist_dir()

        self.foreign_df = self._load_parquet(self.parquet_dir / 'macro/foreign_flow.parquet')
        self.prop_df = self._load_parquet(self.parquet_dir / 'macro/prop_flow.parquet')
        self.price_df = self._load_parquet(self.parquet_dir / 'price/master_price.parquet')

    def _ensure_exist_dir(self):
        self.backtest_dir.mkdir(parents=True, exist_ok=True)

    def _load_parquet(self, path):
        if path.exists():
            try: 
                return pd.read_parquet(path)
            except: 
                print(f"Could NOT read {path}")
                return pd.DataFrame()
        return pd.DataFrame()

    def simulate_trading(self, weights):
        """Mô phỏng 1 vòng giao dịch với bộ trọng số cụ thể"""
        w_poc, w_dtl, w_smart_money, threshold = weights
        
        # Bước 1: Chấm điểm lại toàn bộ lịch sử dựa trên trọng số mới
        self.signals['sim_score'] = 0
        
        # Mô phỏng tính điểm
        self.signals.loc[self.signals['dist_to_poc'] <= 5.0, 'sim_score'] += w_poc
        self.signals.loc[self.signals['dtl'] > 5.0, 'sim_score'] += w_dtl
        self.signals['sim_score'] += self.signals['sm_base_score'] * w_smart_money
        
        # Bước 2: Lọc các lệnh đạt chuẩn Mua
        buy_signals = self.signals[self.signals['sim_score'] >= threshold].copy()
        
        if buy_signals.empty: 
            return 0.0, 0, 0.0, 0.0 # Fitness, Số lệnh, Win-rate, Avg Return
        
        # Bước 3: Tính toán PnL (Lợi nhuận) giả định giữ hàng T+20 (1 tháng)
        winning_trades = 0
        total_pnl_pct = 0.0
        
        for _, trade in buy_signals.iterrows():
            ticker = trade['ticker']
            buy_date = trade['date']
            buy_price = trade['price']
            
            # Tìm giá sau 20 phiên
            future_df = self.prices[(self.prices['ticker'] == ticker) & (self.prices['time'] > buy_date)].head(20)
            if not future_df.empty:
                sell_price = future_df['close'].iloc[-1]
                pnl = (sell_price - buy_price) / buy_price * 100
                total_pnl_pct += pnl
                if pnl > 0: winning_trades += 1
                
        total_trades = len(buy_signals)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_return = total_pnl_pct / total_trades if total_trades > 0 else 0
        
        # Công thức tính điểm tối ưu (Ví dụ: Ưu tiên Lợi nhuận x Tỷ lệ thắng)
        fitness_score = avg_return * (win_rate / 100)
        
        return fitness_score, total_trades, win_rate, avg_return

    def run_grid_search(self):
        """Quét toàn bộ ma trận không gian tham số"""
        print("\n" + "="*65)
        print(f"🚀 KHỞI ĐỘNG ĐỘNG CƠ TỐI ƯU HÓA GRID SEARCH")
        print("="*65)
        
        # Định nghĩa Không gian tìm kiếm (Search Space)
        poc_weights = [10, 15, 20]             # Thử nghiệm cộng 10, 15, 20 điểm cho POC
        dtl_weights = [5, 10, 15]              # Thử nghiệm cộng 5, 10, 15 điểm cho DTL
        sm_multipliers = [0.5, 1.0, 1.5]       # Thử nghiệm hệ số Smart Money
        thresholds = [70, 75, 80, 85]          # Thử nghiệm các ngưỡng Mua
        
        combinations = list(itertools.product(poc_weights, dtl_weights, sm_multipliers, thresholds))
        print(f"[*] Tổng số kịch bản cần giả lập : {len(combinations)} kịch bản.")
        print(f"[*] Tập dữ liệu Tín hiệu lịch sử : {len(self.signals)} dòng.")
        print("-" * 65)
        
        best_score = -999
        best_params = None
        best_stats = None
        
        # Quét qua từng kịch bản
        for weights in tqdm(combinations, desc="Đang quét ma trận"):
            fitness, trades, wr, ret = self.simulate_trading(weights)
            
            # Đạt tối thiểu 30 lệnh (để tránh nhiễu thống kê) và có điểm Fitness cao nhất
            if trades >= 30 and fitness > best_score:
                best_score = fitness
                best_params = weights
                best_stats = (trades, wr, ret)
                
        # In Báo cáo Tối ưu
        if best_params:
            print("\n" + "="*65)
            print(" 🏆 TÌM THẤY BỘ TRỌNG SỐ TỐI ƯU NHẤT (HOLY GRAIL)")
            print("="*65)
            print(f"[*] Điểm tựa nền POC           : +{best_params[0]} điểm")
            print(f"[*] Điểm Cá mập Nhốt hàng (DTL): +{best_params[1]} điểm")
            print(f"[*] Hệ số nhân Smart Money     : x{best_params[2]}")
            print(f"[*] Ngưỡng kích hoạt Mua       : >= {best_params[3]} điểm")
            print("-" * 65)
            print(f"📊 Thống kê Backtest Quá khứ:")
            print(f"   - Tổng số lệnh phát ra  : {best_stats[0]} lệnh")
            print(f"   - Tỷ lệ Thắng (Win-rate): {best_stats[1]:.1f}%")
            print(f"   - Lợi nhuận kỳ vọng/lệnh: {best_stats[2]:.2f}%")
            print("="*65)
        else:
            print("\n[!] Không tìm thấy cấu hình nào tạo ra lợi nhuận với > 30 lệnh.")
            
        return best_params

    # HÀM CHẠY THỰC TẾ TRÊN DỮ LIỆU PARQUET (REAL DATA EXECUTION)
    def run_optimization_pipeline(self, test_tickers=None):
        """
        Đường ống toàn trình: Đọc Parquet -> Trích xuất Tín hiệu -> Chạy Tối ưu hóa Grid Search
        """
        print("\n" + "="*65)
        print(" 🏭 ĐƯỜNG ỐNG TRÍCH XUẤT DỮ LIỆU THỰC & TỐI ƯU HÓA")
        print("="*65)
        
        df_prices = self.price_df
        df_foreign = self.foreign_df
        df_prop = self.prop_df
        
        # Lọc dữ liệu theo danh sách mã nếu có
        if test_tickers:
            df_prices = df_prices[df_prices['ticker'].isin(test_tickers)].copy()
            print(f"[*] Chế độ Test: Chỉ phân tích {len(test_tickers)} mã cổ phiếu.")
            
        df_prices['time'] = pd.to_datetime(df_prices['time'])

        # Hàm nội bộ tính POC
        def calculate_historical_poc(df_ticker, target_date, lookback=130):
            df_past = df_ticker[df_ticker['time'] <= target_date].tail(lookback).copy()
            if df_past.empty: return 0.0
            df_past['price_bin'] = df_past['close'].round(1)
            vol_profile = df_past.groupby('price_bin')['volume'].sum()
            return float(vol_profile.idxmax()) if not vol_profile.empty else 0.0

        print(f"[*] Đang quét lịch sử để xây dựng Tập tín hiệu (Signals Dataset)...")
        mf_engine = MarketFlowAnalyzer()
        signals_data = []

        # Nhóm theo từng mã cổ phiếu để quét các điểm kích hoạt (Trigger Points)
        for ticker, df_t in tqdm(df_prices.groupby('ticker'), desc="Quét các mã"):
            df_t = df_t.sort_values('time').reset_index(drop=True)
            df_t['ema89'] = df_t['close'].ewm(span=89, adjust=False).mean()
            
            df_f_t = df_foreign[df_foreign['ticker'] == ticker] if df_foreign is not None else None
            df_pr_t = df_prop[df_prop['ticker'] == ticker] if df_prop is not None else None
            
            df_t['vol_ma20'] = df_t['volume'].rolling(20).mean()
            # Trigger: Giá cắt lên EMA89 kèm thanh khoản nổ
            trigger_mask = (df_t['close'] > df_t['ema89']) & (df_t['close'].shift(1) <= df_t['ema89'].shift(1)) & (df_t['volume'] > df_t['vol_ma20'])
            trigger_dates = df_t[trigger_mask]['time'].tolist()
            
            for t_date in trigger_dates:
                # Đo lường tồn kho Point-in-time
                mf_res = mf_engine.analyze_flow(
                    ticker=ticker, 
                    df_p=df_t, df_f=df_f_t, df_pr=df_pr_t, 
                    target_date_str=t_date.strftime('%Y-%m-%d')
                )
                
                poc = calculate_historical_poc(df_t, t_date)
                current_price = df_t[df_t['time'] == t_date]['close'].values[0]
                dist_to_poc = ((current_price - poc) / poc * 100) if poc > 0 else 999
                
                signals_data.append({
                    'date': t_date,
                    'ticker': ticker,
                    'price': current_price,
                    'dist_to_poc': dist_to_poc,
                    'dtl': mf_res.get('dtl_days', 0),
                    'sm_base_score': 60  # Điểm cơ sở giả định
                })

        df_signals = pd.DataFrame(signals_data)
        print(f"[*] Đã rút trích thành công {len(df_signals)} điểm kích hoạt từ quá khứ.")

        # Khởi chạy Optimizer nếu có dữ liệu
        if not df_signals.empty:
            df_prices_opt = df_prices[['time', 'ticker', 'close']].copy()
            
            optimizer = QuantOptimizer(historical_signals=df_signals, historical_prices=df_prices_opt)
            best_weights = optimizer.run_grid_search()
            
            # Lưu cache tập tín hiệu
            cache_path = self.backtest_dir / 'historical_signals_cache.parquet'
            df_signals.to_parquet(cache_path, index=False)
            print(f"\n[💾] Đã lưu cache tập tín hiệu tại '{cache_path}'")
            
            return best_weights
        else:
            print("[!] Không tìm thấy bất kỳ tín hiệu nào trong quá khứ để tối ưu.")
            return None
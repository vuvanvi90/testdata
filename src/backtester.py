import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class VectorizedBacktester:
    def __init__(self, price_path='data/parquet/price/master_price.parquet', start_date='2021-01-01', end_date=None):
        self.price_path = Path(price_path)
        self.initial_capital = 1_000_000_000  # Vốn khởi điểm: 1 Tỷ VND
        self.risk_per_trade = 0.02  # Rủi ro 2% vốn cho mỗi lệnh (Stoploss)
        self.lookback = 30

        # Khởi tạo khung thời gian Backtest
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else datetime.now()

        # Thêm danh sách đen vào cấu hình Backtester
        self.ignore_tickers = ['ABR', 'ACG', 'ADP', 'AFX', 'ANT', 'ACL', 'CTR', 'DSC', 'TCI', 'TDP'] 
        
    def load_and_prep_data(self):
        print("--- 1. TẢI DỮ LIỆU VÀ TÍNH TOÁN CHỈ BÁO SIÊU TỐC ---")
        df = pd.read_parquet(self.price_path)
        
        # Loại bỏ các mã lỗi khỏi ma trận dữ liệu
        if self.ignore_tickers:
            df = df[~df['ticker'].isin(self.ignore_tickers)]
        
        # Chuẩn hóa thời gian cổ phiếu
        if pd.api.types.is_numeric_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], unit='ms').dt.normalize()
        else:
            df['time'] = pd.to_datetime(df['time']).dt.normalize()
            
        # LƯU Ý: Không cắt dữ liệu ở đây nữa để các đường EMA89 có đủ lịch sử tính toán (Warm-up)
        # df = df[df['time'] >= '2021-01-01'].copy()
        df = df.sort_values(['ticker', 'time']).reset_index(drop=True)
        
        # ==========================================================
        # ĐỌC VN-INDEX THẬT TỪ FILE PARQUET ĐỂ LÀM BỘ LỌC
        # ==========================================================
        try:
            vnindex_path = Path('data/parquet/macro/vnindex.parquet')
            if vnindex_path.exists():
                df_vni = pd.read_parquet(vnindex_path)
                
                if pd.api.types.is_numeric_dtype(df_vni['time']):
                    df_vni['time'] = pd.to_datetime(df_vni['time'], unit='ms').dt.normalize()
                else:
                    df_vni['time'] = pd.to_datetime(df_vni['time']).dt.normalize()
                    
                df_vni = df_vni.sort_values('time').drop_duplicates('time', keep='last')
                
                # Tính EMA89 chuẩn của VN-Index
                df_vni['vni_ema89'] = df_vni['close'].ewm(span=89, adjust=False).mean()
                df_vni = df_vni.rename(columns={'close': 'vni_close'})
                
                # Ghép VNI vào ma trận giá cổ phiếu
                df = pd.merge(df, df_vni[['time', 'vni_close', 'vni_ema89']], on='time', how='left')
                
                # Điền khuyết (Fill NA) nếu có ngày lệch giữa VNI và Cổ phiếu
                df['vni_close'] = df.groupby('ticker')['vni_close'].ffill()
                df['vni_ema89'] = df.groupby('ticker')['vni_ema89'].ffill()
                
                print("   [*] Đã nạp thành công dữ liệu VN-INDEX THẬT để làm rơ-le Vĩ mô.")
            else:
                print("   [!] Không tìm thấy vnindex.parquet, Backtest chạy chay.")
        except Exception as e:
            print(f"   [!] Lỗi nạp VN-Index: {e}")
            
        return df

    def generate_signals(self, df):
        """Tính toán Wyckoff 2 chiều cho toàn thị trường cùng lúc"""
        print("--- 2. QUÉT TÍN HIỆU WYCKOFF (VECTORIZED) ---")

        # Đếm "tuổi đời" (số ngày giao dịch) của từng cổ phiếu tính đến ngày hiện tại
        df['trading_days'] = df.groupby('ticker').cumcount() + 1
        
        df['vol_ma20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
        df['rel_vol'] = df['volume'] / df['vol_ma20']
        df['ema34'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=34, adjust=False).mean())
        df['ema89'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=89, adjust=False).mean())
        
        df['prev_close'] = df.groupby('ticker')['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df.groupby('ticker')['tr'].transform(lambda x: x.rolling(14).mean())
        
        df['support'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(self.lookback).min().shift(2))
        df['resist'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(self.lookback).max().shift(2))
        
        df['prev_low'] = df.groupby('ticker')['low'].shift(1)
        df['prev_high'] = df.groupby('ticker')['high'].shift(1)
        
        df['signal'] = 'HOLD'
        
        # --- CÔNG TẮC VĨ MÔ THÔNG MINH ---
        if 'vni_close' in df.columns and 'vni_ema89' in df.columns:
            market_uptrend = df['vni_close'] > df['vni_ema89']
        else:
            market_uptrend = True # Mặc định luôn True nếu không có data VNI
            
        # --- LOGIC MUA ---
        # 1. Đánh Breakout (SOS) CHỈ KHI VN-Index đang Uptrend
        # Tránh các mã non trẻ >= 89 phiên giao dịch
        sos_cond = (df['close'] > df['resist']) & (df['rel_vol'] > 1.2) & market_uptrend & (df['trading_days'] >= 89)
        
        # 2. Đánh Bắt Đáy (SPRING) KHÔNG CẦN VN-Index Uptrend
        # Tránh các mã non trẻ >= 89 phiên giao dịch
        # Vì Spring mạnh nhất là khi rũ bỏ hoảng loạn. Chỉ cần cổ phiếu đó giữ được xu hướng riêng của nó (>ema89)
        spring_cond = (df['prev_low'] < df['support']) & (df['close'] >= df['support']) & (df['rel_vol'] < 1.0) & (df['close'] > df['ema89']) & market_uptrend & (df['trading_days'] >= 89)
        
        df.loc[sos_cond, 'signal'] = 'BUY_SOS'
        df.loc[spring_cond, 'signal'] = 'BUY_SPRING'
        
        # --- LOGIC BÁN ---
        sow_cond = (df['close'] < df['support']) & (df['rel_vol'] > 1.2)
        utad_cond = ((df['prev_high'] > df['resist']) | (df['high'] > df['resist'])) & (df['close'] <= df['resist']) & (df['rel_vol'] > 1.0)
        
        df.loc[sow_cond | utad_cond, 'signal'] = 'SELL_WYCKOFF'

        # ==========================================================
        # CHỐT KHUNG THỜI GIAN BACKTEST (SAU KHI ĐÃ TÍNH XONG CHỈ BÁO)
        # ==========================================================
        df = df[(df['time'] >= self.start_date) & (df['time'] <= self.end_date)].copy()
        
        return df

    def run_simulation(self, df):
        print("--- 3. MÔ PHỎNG GIAO DỊCH VÀ QUẢN TRỊ VỐN ---")
        dates = sorted(df['time'].unique())
        
        capital = self.initial_capital
        portfolio = {} # {ticker: {'shares': x, 'entry': y, 'sl': z}}
        
        equity_curve = []
        trade_log = []
        
        # Đưa dataframe vào dạng Dict 2 chiều (Ngày -> Mã) để truy xuất O(1)
        # Bỏ qua các dòng bị lỗi thiếu giá trị (NaN) ở các cột quan trọng
        df_valid = df.dropna(subset=['close', 'atr', 'signal']).copy()
        
        daily_data = {}
        for d, group in df_valid.groupby('time'):
            daily_data[d] = group.set_index('ticker').to_dict('index')

        print(f"   -> Đang chạy loop qua {len(dates)} ngày giao dịch...")
        
        for d in dates:
            if d not in daily_data: continue
            today_market = daily_data[d]
            
            # 1. CẬP NHẬT GIÁ TRỊ VÀ KIỂM TRA BÁN (SL / TRAILING STOP / WYCKOFF)
            tickers_to_sell = []
            for ticker, pos in portfolio.items():
                if ticker in today_market:
                    row = today_market[ticker]
                    curr_price = row['close']
                    signal = row['signal']
                    atr = row['atr']
                    
                    # --- VŨ KHÍ 2: TRAILING STOP (CHỐT LỜI ĐỘNG) ---
                    # Nếu giá hôm nay cao hơn đỉnh cũ đã ghi nhận
                    if curr_price > pos.get('highest_price', pos['entry']):
                        pos['highest_price'] = curr_price
                        # Kéo mức cắt lỗ lên theo đỉnh mới (Bảo vệ lợi nhuận)
                        new_sl = curr_price - (atr * 2.5) 
                        if new_sl > pos['sl']:
                            pos['sl'] = new_sl

                    # Bán nếu chạm Dừng lỗ (hoặc Trailing Stop) hoặc có tín hiệu Xấu
                    if curr_price <= pos['sl'] or signal == 'SELL_WYCKOFF':
                        sell_val = pos['shares'] * curr_price
                        capital += sell_val
                        
                        pnl = (curr_price - pos['entry']) / pos['entry']
                        trade_log.append({'date': d, 'ticker': ticker, 'type': 'SELL', 'pnl_%': pnl*100})
                        tickers_to_sell.append(ticker)

            for t in tickers_to_sell:
                del portfolio[t]

            # 2. KIỂM TRA MUA (Chỉ Mua khi được mở khóa Vĩ mô)
            if len(portfolio) < 5:
                # Lấy danh sách các mã có tín hiệu MUA kèm dữ liệu của nó
                buy_list = [(t, r) for t, r in today_market.items() if 'BUY' in r['signal'] and t not in portfolio]
                
                # Sắp xếp (Ranking): Ưu tiên mã có Dòng tiền đột biến nhất (rel_vol cao nhất)
                buy_list.sort(key=lambda item: item[1]['rel_vol'], reverse=True)
                
                # Lấy ra danh sách ticker đã xếp hạng
                buy_candidates = [item[0] for item in buy_list]
                
                for ticker in buy_candidates:
                    if len(portfolio) >= 5: break
                    row = today_market[ticker]
                    curr_price = row['close']
                    atr = row['atr']
                    
                    sl_price = curr_price - (atr * 2.5)
                    risk_amount = curr_price - sl_price
                    
                    if risk_amount > 0:
                        # Tính số lượng cổ phiếu theo mức rủi ro 2%
                        shares = int((capital * self.risk_per_trade) / risk_amount)
                        
                        # Ràng buộc tỷ trọng: Không mua quá 20% tổng vốn cho 1 mã
                        max_capital_per_trade = capital * 0.20
                        if (shares * curr_price) > max_capital_per_trade:
                            shares = int(max_capital_per_trade / curr_price)
                            
                        shares = (shares // 100) * 100 # Làm tròn lô 100
                        cost = shares * curr_price
                        
                        if cost <= capital and shares > 0:
                            capital -= cost
                            portfolio[ticker] = {
                                'shares': shares, 
                                'entry': curr_price, 
                                'sl': sl_price,
                                'highest_price': curr_price # Khởi tạo đỉnh ban đầu
                            }
                            trade_log.append({'date': d, 'ticker': ticker, 'type': 'BUY', 'price': curr_price})

            # 3. GHI NHẬN TÀI SẢN (EQUITY CURVE)
            total_value = capital
            for ticker, pos in portfolio.items():
                if ticker in today_market:
                    total_value += pos['shares'] * today_market[ticker]['close']
                else:
                    total_value += pos['shares'] * pos['entry'] # Nếu mã bị hủy niêm yết hoặc không có giá hôm nay
            
            equity_curve.append({'date': d, 'equity': total_value})

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_log)

    def calculate_metrics(self, equity_df, trades_df):
        start_str = self.start_date.strftime('%d/%m/%Y')
        end_str = self.end_date.strftime('%d/%m/%Y')
        
        print("\n" + "="*50)
        print(f" 📊 KẾT QUẢ BACKTEST ({start_str} - {end_str})")
        print("="*50)
        
        initial = self.initial_capital
        final = equity_df.iloc[-1]['equity']
        net_profit = final - initial
        
        # 1. Lợi nhuận gộp hàng năm (CAGR)
        days = (equity_df.iloc[-1]['date'] - equity_df.iloc[0]['date']).days
        years = days / 365.25
        cagr = (final / initial) ** (1 / years) - 1
        
        # Phương trình Toán học của CAGR được tính như sau:
        # $CAGR = \left( \frac{Ending Value}{Beginning Value} \right)^{\frac{1}{n}} - 1$
        
        # 2. Max Drawdown (Cú sụt giảm tài khoản lớn nhất)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_dd = equity_df['drawdown'].min()
        
        # 3. Tỷ lệ Thắng (Winrate)
        sells = trades_df[trades_df['type'] == 'SELL']
        if not sells.empty:
            wins = len(sells[sells['pnl_%'] > 0])
            losses = len(sells[sells['pnl_%'] <= 0])
            winrate = wins / len(sells)
        else:
            winrate = 0
            
        print(f"💰 Vốn ban đầu:      {initial:,.0f} đ")
        print(f"💵 Vốn cuối kỳ:      {final:,.0f} đ")
        print(f"📈 Lãi ròng:         {net_profit:,.0f} đ")
        print(f"🚀 CAGR (Trung bình):{cagr*100:.2f}% / năm")
        print(f"⚠️ Max Drawdown:     {max_dd*100:.2f}% (Mức chịu đựng rủi ro)")
        print(f"🎯 Tỷ lệ Thắng (WR): {winrate*100:.2f}% (Tổng {len(sells)} lệnh)")
        print("="*45)
        
        # Vẽ biểu đồ Equity Curve
        # plt.figure(figsize=(12, 6))
        # plt.plot(equity_df['date'], equity_df['equity'], label='Danh mục Wyckoff', color='blue')
        # plt.title('Mô phỏng Tăng trưởng Tài khoản (Equity Curve)')
        # plt.xlabel('Thời gian')
        # plt.ylabel('VND')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

if __name__ == "__main__":
    # Yêu cầu cài đặt: pip install matplotlib
    bt = VectorizedBacktester()
    df_market = bt.load_and_prep_data()
    df_signals = bt.generate_signals(df_market)
    
    eq_curve, trades = bt.run_simulation(df_signals)
    bt.calculate_metrics(eq_curve, trades)
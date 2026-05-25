import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

class MarketScreener:
    """
    ĐÀI QUAN SÁT VĨ MÔ & QUÉT SIÊU CỔ PHIẾU (MARKET SCREENER) V1.0
    🚀 Top-down Screener: Exact Active Matching, VWAP Distance, RS Score & TQS.
    """
    def __init__(self, data_dir='data/parquet'):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Hệ thống Radar Quét Toàn Thị trường...")
        self.data_dir = Path(data_dir)
        self.DIVISOR = 1_000_000_000 # Quy đổi ra Tỷ VNĐ
        
        self._load_data()
        self._build_mappings()

    def _load_data(self):
        """Nạp dữ liệu EOD làm Chân lý tuyệt đối"""
        def load_pq(sub_path):
            p = self.data_dir / sub_path
            return pd.read_parquet(p) if p.exists() else pd.DataFrame()

        self.df_price = load_pq('price/master_price_l2.parquet')
        self.df_prop = load_pq('macro/prop_flow.parquet')
        self.df_idx = load_pq('macro/index_components.parquet')

        if not self.df_price.empty and 'matched_volume' in self.df_price.columns and 'volume' not in self.df_price.columns:
            self.df_price = self.df_price.rename(columns={'matched_volume': 'volume'})

        # Xử lý chuẩn hóa trục thời gian
        for df in [self.df_price, self.df_prop]:
            if not df.empty:
                if pd.api.types.is_numeric_dtype(df['time']):
                    df['date'] = pd.to_datetime(df['time'], unit='ms').dt.date
                else:
                    df['date'] = pd.to_datetime(df['time']).dt.date

    def _build_mappings(self):
        """Tạo Mapping để cô lập Rổ Cổ Phiếu (Universe Isolation)"""
        self.ticker_to_universe = {}
        if not self.df_idx.empty:
            for _, row in self.df_idx.iterrows():
                ticker, idx_code = row['ticker'], row['index_code']
                if ticker not in self.ticker_to_universe or idx_code in ['VN30', 'VNMidCap', 'VNSmallCap']:
                    self.ticker_to_universe[ticker] = idx_code

    def run_screener(self, target_universes=['VN30', 'VNMidCap'], start_date=None, end_date=None, lookback_days=30):
        """Thực thi quét thị trường theo Khung thời gian và Rổ tùy chỉnh."""
        if self.df_price.empty:
            print("[!] Không có dữ liệu master_price_l2.")
            return pd.DataFrame()

        df = self.df_price.copy()
        
        # 1. XÁC ĐỊNH TRỤC THỜI GIAN ĐỘNG
        all_dates = sorted(df['date'].dropna().unique())
        if not all_dates: return pd.DataFrame()
        
        if end_date is None:
            end_date = all_dates[-1]
        else:
            end_date = pd.to_datetime(end_date).date()
            
        if start_date is None:
            valid_past_dates = [d for d in all_dates if d <= end_date]
            start_date = valid_past_dates[-lookback_days] if len(valid_past_dates) >= lookback_days else valid_past_dates[0]
        else:
            start_date = pd.to_datetime(start_date).date()

        # 2. LỌC DỮ LIỆU THỜI GIAN VÀ RỔ UNIVERSE
        df['universe'] = df['ticker'].map(lambda x: self.ticker_to_universe.get(x, 'UNKNOWN'))
        
        df_period = df[(df['date'] >= start_date) & 
                       (df['date'] <= end_date) & 
                       (df['universe'].isin(target_universes))].copy()
        
        if df_period.empty: return pd.DataFrame()

        # 3. GỘP DỮ LIỆU TỰ DOANH ĐỂ TÍNH SHADOW FLOW
        if not self.df_prop.empty:
            prop_cols = ['ticker', 'date', 'prop_buy_val_deal', 'prop_sell_val_deal']
            prop_merge = self.df_prop[[c for c in prop_cols if c in self.df_prop.columns]]
            df_period = pd.merge(df_period, prop_merge, on=['ticker', 'date'], how='left')
        
        df_period.fillna(0, inplace=True)

        # 4. TÍNH TOÁN CÁC METRICS CHO TỪNG MÃ (CHÍNH XÁC THEO DATA CONTRACT)
        results = []
        grouped = df_period.groupby('ticker')
        
        for ticker, group in grouped:
            group = group.sort_values('date')
            if len(group) < (lookback_days * 0.5): continue 
            
            # --- Biến động Giá ---
            start_p = float(group['close'].iloc[0])
            end_p = float(group['close'].iloc[-1])
            price_change = ((end_p - start_p) / start_p) * 100 if start_p > 0 else 0
            
            # --- Thanh khoản & VWAP ---
            total_vol = float(group['volume'].sum())
            total_val_bn = float(group['matched_value'].sum()) / self.DIVISOR
            
            # VWAP = Tổng Giá trị Khớp / Tổng Volume Khớp
            vwap_price = (total_val_bn * self.DIVISOR) / total_vol if total_vol > 0 else end_p
            vwap_dist = ((end_p - vwap_price) / vwap_price) * 100 if vwap_price > 0 else 0
            
            # CHUYỂN SANG ĐO LƯỜNG ÁP LỰC CUNG CẦU SỔ LỆNH (ORDER BOOK PRESSURE)
            # Dùng Tổng Đặt Mua và Tổng Đặt Bán để đo lường ý chí thị trường
            total_buy_intent = float(group['total_buy_trade_volume'].sum())
            total_sell_intent = float(group['total_sell_trade_volume'].sum())
            
            total_intent = total_buy_intent + total_sell_intent
            # Tỷ lệ Lực Cầu (Buy Intent Ratio): Ý chí muốn gom hàng của toàn thị trường
            buy_intent_ratio = (total_buy_intent / total_intent * 100) if total_intent > 0 else 50.0
            
            # --- Shadow Flow (Gom Ngầm Thỏa thuận) ---
            # Ngoại Thỏa thuận Ròng = Mua - Bán
            f_deal_net = (group['fr_buy_value_deal'].sum() - group['fr_sell_value_deal'].sum()) / self.DIVISOR
            # Tự doanh Thỏa thuận Ròng = Mua - Bán
            p_deal_net = (group.get('prop_buy_val_deal', pd.Series(0)).sum() - group.get('prop_sell_val_deal', pd.Series(0)).sum()) / self.DIVISOR
            
            total_deal_val = float(group['deal_value'].sum()) / self.DIVISOR
            
            # Lái Nội ẩn mình = Tổng Thỏa thuận - |Ngoại Ròng| - |Tự doanh Ròng|
            shadow_flow = max(0, total_deal_val - abs(f_deal_net) - abs(p_deal_net))
            
            results.append({
                'ticker': ticker,
                'universe': group['universe'].iloc[0],
                'close': end_p,
                'change_pct': price_change,
                'vwap_dist_pct': vwap_dist,
                'active_buy_pct': buy_intent_ratio,
                'shadow_bn': shadow_flow,
                'total_val_bn': total_val_bn
            })
            
        res_df = pd.DataFrame(results)
        if res_df.empty: return res_df
        
        # 5. CHẤM ĐIỂM SỨC MẠNH TƯƠNG ĐỐI (RS 1-99)
        res_df['RS_Score'] = res_df['change_pct'].rank(pct=True) * 99
        
        # 6. CHẤM ĐIỂM CHẤT LƯỢNG SÓNG (TQS - Trend Quality Score)
        norm_buy = (res_df['active_buy_pct'] - 0) / (100 - 0) * 100
        
        # Hàm tính điểm VWAP (Cách VWAP dưới 5% là an toàn, cách quá xa bị trừ điểm mạnh)
        res_df['vwap_score'] = 100 - (res_df['vwap_dist_pct'].abs() * 2) 
        res_df['vwap_score'] = res_df['vwap_score'].clip(lower=0, upper=100)
        
        res_df['TQS'] = (res_df['RS_Score'] * 0.40) + \
                        (norm_buy * 0.30) + \
                        (res_df['vwap_score'] * 0.20) + \
                        (np.where(res_df['shadow_bn'] > 0, 10, 0))
        
        # 7. ĐÁNH NHÃN VPA (Volume Price Analysis)
        median_val = res_df['total_val_bn'].median()
        
        conditions = [
            (res_df['change_pct'] < -2) & (res_df['active_buy_pct'] > 55), 
            (res_df['change_pct'] > 5)  & (res_df['total_val_bn'] < median_val * 0.5), 
            (res_df['change_pct'] > 5)  & (res_df['active_buy_pct'] > 60) 
        ]
        choices = [
            '🛡️ ĐẠP GOM (Gom vùng giá thấp)', # Lực kê Mua đỡ giá rất lớn nhưng Giá vẫn bị ép giảm
            '💨 CẠN CUNG (Markup Squeeze)', # Tăng mạnh nhưng Thanh khoản rất cạn
            '🚀 SÓNG HỮU CƠ (Organic Trend)' # Tăng mạnh + Lực kê Mua áp đảo
        ]
        res_df['VPA_Tag'] = np.select(conditions, choices, default='-')
        
        return res_df.sort_values('TQS', ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    screener = MarketScreener()
    
    # Kịch bản 1: Quét độc lập VN30 & MidCap trong 30 ngày qua
    lookback = 10
    # target_baskets = ['VN30', 'VNMidCap']
    target_baskets = ['VN30']
    
    print("\n" + "="*95)
    print(f" 📡 ĐANG QUÉT TÍN HIỆU TOÀN THỊ TRƯỜNG (RỔ: {', '.join(target_baskets)} | KHUNG: {lookback}D)")
    print("="*95)
    
    df_report = screener.run_screener(target_universes=target_baskets, lookback_days=lookback)
    
    if not df_report.empty:
        print("\n🏆 TOP 10 MÃ DẪN DẮT (TQS CAO NHẤT) - KHUYẾN NGHỊ ĐƯA VÀO SNIPER:")
        print(f"{'MÃ':<5} | {'RỔ':<10} | {'TĂNG GIÁ':<10} | {'RS':<5} | {'TQS':<5} | {'MUA C.ĐỘNG':<12} | {'GHI CHÚ VPA'}")
        print("-" * 95)
        for _, r in df_report.head(10).iterrows():
            print(f"{r['ticker']:<5} | {r['universe']:<10} | {r['change_pct']:>6.2f}%   | {r['RS_Score']:>4.0f} | {r['TQS']:>4.0f} | {r['active_buy_pct']:>8.1f}%   | {r['VPA_Tag']}")
            
        print("\n" + "-" * 95)
        print("🚨 TOP 10 MÃ SUY YẾU CẦN TRÁNH (RS THẤP NHẤT):")
        df_laggards = df_report.sort_values('RS_Score', ascending=True).head(10)
        for _, r in df_laggards.iterrows():
            print(f"{r['ticker']:<5} | {r['universe']:<10} | {r['change_pct']:>6.2f}%   | {r['RS_Score']:>4.0f} | {r['TQS']:>4.0f} | {r['active_buy_pct']:>8.1f}%   | {r['VPA_Tag']}")
            
        print("="*95)
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from config import settings

# --- DANH SÁCH MÙNG 1 TẾT NGUYÊN ĐÁN (2021-2030) ---
# Dùng để tính toán thời điểm "Tháng Chạp"
LUNAR_NEW_YEARS = [
    '2021-02-12', '2022-02-01', '2023-01-22', '2024-02-10', 
    '2025-01-29', '2026-02-17'
]

class WyckoffForecaster:
    def __init__(self, price_dir=None, output_dir=None, run_date=datetime.now()):
        self.price_dir = Path(price_dir) if price_dir else Path('data/parquet/price/master_price.parquet')
        self.output_dir = Path(output_dir) if output_dir else Path('data/forecast')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lookback = settings.LOOKBACK_TR
        self.run_date = pd.to_datetime(run_date)

    def _is_pre_tet_period(self, current_date):
        """Kiểm tra xem ngày hiện tại có nằm trong 30 ngày trước Tết Âm lịch không"""
        current_date_str = current_date.strftime('%Y-%m-%d')
        for tet in LUNAR_NEW_YEARS:
            tet_date = pd.to_datetime(tet)
            # Kiểm tra: Trong vòng 30 ngày trước Tết
            if 0 < (tet_date - current_date).days <= 30:
                return True
        return False

    def _calculate_indicators(self, df):
        """Bước 1: Phân tích Khối lượng và Giá (VPA)"""
        df['spread'] = df['high'] - df['low']
        df['vol_ma20'] = df['volume'].rolling(window=settings.VOL_MA_PERIOD).mean()
        df['rel_vol'] = df['volume'] / df['vol_ma20']
        
        # EMA
        df['ema34'] = df['close'].ewm(span=settings.EMA_FAST, adjust=False).mean()
        df['ema89'] = df['close'].ewm(span=settings.EMA_SLOW, adjust=False).mean()
        
        # --- NEW: Tính ATR (Average True Range) để dùng cho Stoploss động ---
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        # Phân loại Volume
        df['vol_type'] = np.where(df['rel_vol'] > 2.0, 'Ultra High', 
                         np.where(df['rel_vol'] > 1.5, 'High', 
                         np.where(df['rel_vol'] < 0.6, 'Low', 'Normal')))
        return df

    def _detect_signals(self, df):
        """Bước 2: Nhận diện tín hiệu Wyckoff (2 Chiều: Mua & Bán)"""
        if len(df) < (self.lookback + 2):
            return "NEUTRAL", "Không đủ dữ liệu.", 0, 0

        range_df = df.iloc[-(self.lookback + 2) : -2] 
        sup, res = range_df['low'].min(), range_df['high'].max()
        
        curr, prev = df.iloc[-1], df.iloc[-2]
        curr_close, curr_vol, curr_rel_vol = curr['close'], curr['volume'], curr['rel_vol']
        prev_low, prev_high = prev['low'], prev['high']
        
        signal = "NEUTRAL"
        note = "Giá đang biến động trong vùng tích lũy."

        # Kiểm tra hiệu ứng mùa vụ (nếu có giữ logic cũ)
        is_pre_tet = self._is_pre_tet_period(self.run_date) if hasattr(self, '_is_pre_tet_period') else False
        sos_vol_req = 2.0 if is_pre_tet else 1.2
        
        # ==========================================
        # 🟢 CHIỀU MUA (ACCUMULATION)
        # ==========================================
        # 1. SPRING (Cú rũ bỏ)
        if (prev_low < sup) and (curr_close >= sup):
            trend_ok = (curr_close > curr['ema89']) or (curr_close > curr['ema34'] * 0.98)
            if curr_rel_vol < 1.0 and trend_ok:
                signal = "SPRING"
                note = f"SPRING: Rũ bỏ hỗ trợ {sup:,.0f} (Vol thấp). Trend ổn."

        # 2. SOS (Dấu hiệu sức mạnh)
        elif curr_close > res:
            if curr_rel_vol > sos_vol_req:
                signal = "SOS"
                note = f"SOS: Bứt phá kháng cự {res:,.0f}. RelVol: {curr_rel_vol:.2f}"

        # ==========================================
        # 🔴 CHIỀU BÁN (DISTRIBUTION)
        # ==========================================
        # 3. UTAD (Bẫy Tăng Giá / Upthrust)
        # Điều kiện: Nến trước (hoặc trong phiên) chọc thủng Kháng cự, nhưng nến hiện tại đóng cửa dưới Kháng cự.
        elif (prev_high > res or curr['high'] > res) and (curr_close <= res):
            if curr_rel_vol > 1.0: # Volume lớn cho thấy lực bán dội xuống mạnh
                signal = "UTAD"
                note = f"UTAD (Bẫy Bò): Kéo vượt {res:,.0f} nhưng bị bán ngược. Rủi ro đảo chiều."

        # 4. SOW / MARKDOWN (Dấu hiệu suy yếu / Gãy nền)
        elif curr_close < sup:
            if curr_rel_vol > 1.2:
                signal = "SOW"
                note = f"SOW: Gãy hỗ trợ {sup:,.0f} với Volume lớn ({curr_rel_vol:.2f}x). Xác nhận Downtrend."
            else:
                signal = "MARKDOWN"
                note = f"MARKDOWN: Trôi qua hỗ trợ {sup:,.0f} do cạn Cầu."

        return signal, note, sup, res

    def run_forecast(self):
        results = []
        print(f"--- Bắt đầu dự báo Wyckoff: {self.run_date.strftime('%d/%m/%Y')} ---")

        if not self.price_dir.exists():
            print(f"[!] Không tìm thấy dữ liệu tại: {self.price_dir}")
            return pd.DataFrame()

        # Đọc toàn bộ dữ liệu thị trường chỉ với 1 dòng lệnh
        print("Đang tải dữ liệu Parquet...")
        df_master = pd.read_parquet(self.price_dir)

        # Group by mã cổ phiếu để xử lý
        for ticker, df_ticker in df_master.groupby('ticker'):
            try:
                # Sắp xếp lại thời gian cho chuẩn
                df = df_ticker.sort_values('time').reset_index(drop=True)
                
                df = self._calculate_indicators(df)
                signal, note, sup, res = self._detect_signals(df)
                curr = df.iloc[-1]
                
                # Lấy dữ liệu 250 phiên gần nhất (1 năm)
                recent_250 = df.tail(250)
                high_52w = recent_250['high'].max()
                last_price = curr['close']
                dist_to_sup = ((last_price - sup) / sup) * 100 if sup > 0 else 0

                # Tính khoảng cách tới đỉnh 52 tuần (%)
                dist_to_52w_high = ((last_price - high_52w) / high_52w) * 100

                results.append({
                    'Ticker': ticker,
                    'Price': last_price,
                    'ATR': round(curr['atr'], 2) if not pd.isna(curr['atr']) else 0, # Lưu ATR để Trading dùng
                    'EMA34': round(curr['ema34'], 2),
                    'EMA89': round(curr['ema89'], 2),
                    'Signal': signal,
                    'Support': sup,
                    'Resistance': res,
                    'Dist_to_Support_%': round(dist_to_sup, 4),
                    'Dist_to_52W_High_%': round(dist_to_52w_high, 4),
                    'VPA_Status': curr.get('vol_type', 'Normal'),
                    'Analysis': note
                })
            except Exception as e:
                continue

        report_df = pd.DataFrame(results)
        if not report_df.empty:
            # Xếp hạng ưu tiên: Mua (1, 2) -> Bán (3, 4, 5) -> Đi ngang (6)
            priority_map = {'SPRING': 1, 'SOS': 2, 'UTAD': 3, 'SOW': 4, 'MARKDOWN': 5, 'NEUTRAL': 6}
            report_df['priority'] = report_df['Signal'].map(priority_map)
            report_df = report_df.sort_values('priority').drop(columns=['priority'])
            output_path = self.output_dir / 'wyckoff_forecast.csv'
            report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"[OK] Đã phân tích xong {len(report_df)} mã. Kết quả lưu tại: {output_path}")
        
        return report_df

if __name__ == "__main__":
    forecaster = WyckoffForecaster()
    report = forecaster.run_forecast()